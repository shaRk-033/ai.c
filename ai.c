#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#define M_PI 3.14159265358979323846

struct Config
{
    int batch_size;
    int seq_len;
    int num_heads;
    int d_model;
};

struct Config config = {4, 128, 8, 512};

struct Embedding
{
    float *inp_emb;
    float *pos_emb;
    float *dinp_emb;
    float *dpos_emb;
};

struct AttentionBlock
{
    float *attn_weight;
    float *attn_bias;
    float *proj_weight;
    float *proj_bias;
    float *mask;
    float *dattn_weight;
    float *dattn_bias;
    float *dproj_weight;
    float *dproj_bias;
};

struct MLP
{
    float *c_fc_weight;
    float *c_fc_bias;
    float *c_proj_weight;
    float *c_proj_bias;
    float *d_fc_weight;
    float *d_fc_bias;
    float *d_proj_weight;
    float *d_proj_bias;
};

struct LayerNormBlock
{
    float *ln1_weight;
    float *ln1_bias;
    float *dln1_weight;
    float *dln1_bias;
};

struct DecoderLayer
{
    struct AttentionBlock *causal_attention;
    struct MLP *mlp;
    struct LayerNormBlock *ln1;
    struct LayerNormBlock *ln2;
};

struct gpt
{
    struct Embedding *embedding;
    struct DecoderLayer *decoder_layer;
};

// Box Muller transform
float randn()
{
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    return sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
}

void relu(float *x, int size)
{
    for (int i = 0; i < size; i++)
    {
        x[i] = x[i] > 0 ? x[i] : 0;
    }
}

void matmul(const float *inp, const float *weight, float *bias, float *out, int m, int k, int n, int B)
{

#pragma omp parallel for collapse(2)
    for (int batch = 0; batch < B; batch++)
    {
        for (int i = 0; i < m; i++)
        {
            int b_i = batch * m * n + i * m;
            for (int l = 0; l < k; l++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (l == 0)
                    {
                        out[b_i + j] = bias == NULL ? 0.0f : bias[j];
                    }
                    out[b_i + j] += inp[batch * m * k + i * m + l] * weight[l * n + j];
                }
            }
        }
    }
}
// out = inp @ weight + bias--> dinp = dout @ (weight).T
// dweight = (dinp).T @ dout, dbias = dout.sum()
// Transposing and matrix mul gets lil bit hairy, i tried on a napkin first
void matmul_backprop(const float *inp, const float *weight, const float *dout, float *din, float *dweight, float *dbias, int m, int k, int n, int B)
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < B; i++)
    {
        for (int j = 0; j < m; j++)
        {
            for (int l = 0; l < n; l++)
            {
                for (int x = 0; x < k; x++)
                {
                    if (x == 0)
                    {
                        dbias[l] += dout[i * m * n + j * n + l];
                    }
                    din[i * m * k + j * k + x] += dout[i * m * n + j * n + l] * weight[x * n + l];
                }
            }
            for(int x = 0; x < k; x++){
                for(int l = 0; l < n; l++){
                    dweight[x * n + l] += inp[i * m * k + j * k + x] * dout[i * m * n + j * n + l];
                }
            }
        }
    }
}
// Softmax = log(exp(logits)/sum(exp(logits))) = logits - log(sum(exp(logits)))
void Softmax(float *logits, float *out, int B, int T, int C)
{
    for (int i = 0; i < B * T; i++)
    {
        float maxi = -INFINITY; // numerical stability
        for (int j = 0; j < C; j++)
        {
            maxi = fmax(maxi, logits[i * C + j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < C; j++)
        {
            sum += exp(logits[i * C + j] - maxi);
        }
        for (int j = 0; j < C; j++)
        {
            out[i * C + j] = logits[i * C + j] - maxi - log(sum);
        }
    }
}

float CrossEntropy(float *logits, int *targets, int B, int T, int C)
{
    float loss = 0.0f;
    float *probs = (float *)calloc(B * T, sizeof(float));
    Softmax(logits, probs, B, T, C);
    for (int i = 0; i < B * T; i++)
    {
        loss -= probs[i * C + targets[i]];
    }
    free(probs);
    return loss / (B * T);
}

void embeddings_layer(const int *inp, const int *inp_emb, const float *pos_emb, float *out, int B, int T, int C)
{
    for (int i = 0; i < B; i++)
    {
        for (int j = 0; j < T; j++)
        {
            for (int k = 0; k < C; k++)
            {
                out[i * T * C + j * C + k] = inp_emb[inp[i * T + j] * C + k] + pos_emb[j * C + k];
            }
        }
    }
}

void layernorm(const float *inp, float *cache_mean, float *cache_var, const float *scale_weights, const float *shift_bias, float *out, int B, int T, int C)
{
    float epsilon = 1e-5;
    float *mean = (float *)calloc(B * T, sizeof(float));
    float *var = (float *)calloc(B * T, sizeof(float));
    float c = C;
    for (int i = 0; i < B * T; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < C; j++)
        {
            sum += inp[i * C + j];
        }
        mean[i] = sum / c;
        float m = mean[i];
        float v = 0.0f;
        for (int j = 0; j < C; j++)
        {
            v += (inp[i * C + j] - m) * (inp[i * C + j] - m);
        }
        var[i] = sqrt(v / (c - 1) + epsilon);
        cache_mean[i] = 0.9f * cache_mean[i] + 0.1f * mean[i];
        cache_var[i] = 0.9f * cache_var[i] + 0.1f * var[i];
        for (int j = 0; j < C; j++)
        {
            out[i * C + j] = scale_weights[j] * (inp[i * C + j] - mean[i]) / var[i] + shift_bias[j];
        }
    }
    free(mean);
    free(var);
}

float *CausalAttention(const float *input, float *output, struct AttentionBlock attn, int B, int T, int C, int num_heads)
{
    assert(C % num_heads == 0);
    int head_size = C / num_heads;
    // --------- Stage 1-----------
    // inp @ qkv = B, T, C @ C, 3C = B, T, 3C
    // --------- Stage 2-----------
    // Q @ K.T = B, N, T, H @ B, N, H, T = (B, N, T, T)/sqrt(H)
    // self attention = apply mask and softmax
    // self attention  @ V = B, N, T, T @ B, N, T, H = B, N, T, H = attention score
    // --------- Stage 3-----------
    // attention score transpose(1,2) = B, T, N, H
    // B, T, N, H @ C, C + 1,C = B, T, C--->output
    float *qkv = (float *)calloc(B * T * num_heads * 3 * head_size, sizeof(float));
    float *attn_raw = (float *)calloc(B * T * num_heads * T, sizeof(float));
    // Stage 1
    matmul(input, attn.attn_weight, attn.attn_bias, qkv, T, C, 3 * C, B);
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < B; i++)
    {
        for (int j = 0; j < T; j++)
        {
            // Self attention and masking, Stage 2
            // q @ k.T = N, T, H @ N, H, T = N, T, T
            for (int x = 0; x < num_heads; x++)
            {
                float *query = qkv + i * T * 3 * C + j * 3 * C;
                float norm = 1.0f / sqrt(head_size);
                float val = 0.0f;
                float maxi = -INFINITY;
                for (int t = 0; t < T; t++)
                { // This loop ensures we dont look at future tokens
                    if(t <= j){
                        float *key = qkv + i * T * 3 * C + t * 3 * C + C;
                        for (int y = 0; y < head_size; y++)
                        {
                            val += query[x * head_size + y] * key[x * head_size + y];
                        }
                        val *= norm;
                        attn_raw[i * num_heads * T * T + x * T * T + j * T + t] = val;
                        maxi = maxi>val?maxi:val;
                    }
                    else{
                        attn_raw[i * num_heads * T * T + x * T * T + j * T + t] = -INFINITY;
                    }
                }

                // softmax, copypasta
                
                float sum = 0.0f;
                for (int t = 0; t < T; t++)
                {
                    sum += exp(attn_raw[i * num_heads * T * T + x * T * T + j * T + t] - maxi);
                }
                int ind = i * num_heads * T * T + x * T * T + j * T;
                for (int t = 0; t < T; t++)
                {
                    if(t<=j){attn_raw[ind + t] = attn_raw[ind + t] - maxi - log(sum);}
                    else{
                        attn_raw[ind + t] = 0;
                    }
                }
                // Stage 3
                for (int t = 0; t <= j; t++)
                {
                    float *value = qkv + i * T * 3 * C + t * 3 * C + 2 * C;
                    for (int y = 0; y < head_size; y++)
                    {
                        output[i * T * C + j * C + x * head_size + y] = attn_raw[i * num_heads * T * T + x * T * T + j * T + t] * value[x * head_size + y];
                    }
                }
            }
        }
    }
    return attn_raw;
    // one of the most annoying and fun things to code
}

void causal_attention_backprop(float *output, float *input, float *dout, float *din, float *dattn_weights, float *dattn_bias, int B, int T, int C, int num_heads){
     
}

void layernorm_backprop(float *output, float *input, float *dout, float *din, float *dscale_weights, float *dshift_bias, int B, int T, int C){
     
}

void loss_backward(float *logits, int B, int T, int C)
{
    float *dlogits = (float *)calloc(B * T * C, sizeof(float));
    Softmax(logits, dlogits, B, T, C);
    for (int i = 0; i < B * T; i++)
    {
        for (int j = 0; j < C; j++)
        {
            dlogits[i * C + j] -= 1.0f;
        }
    }
}


