#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#ifdef _OPENMP
#include <omp.h>
#endif

struct Config
{
    int batch_size;
    int seq_len;
    int num_heads;
    int d_model;
    int vocab_size;
    int n_layers;
};

struct Config config = {4, 128, 8, 512};

struct Embedding
{
    float *input;
    float *pos_emb;
    float *inp_emb;
    float *dinp_emb;
    float *dpos_emb;
    float *output;
    float *doutput;
};

struct AttentionBlock
{
    float *input;
    float *attn_weight;
    float *attn_bias;
    float *proj_weight;
    float *proj_bias;
    float *dattn_weight;
    float *dattn_bias;
    float *dproj_weight;
    float *dproj_bias;
    float *output;
    float *doutput;
};

struct MLP
{
    float *input;
    float *c_fc_weight;
    float *c_fc_bias;
    float *c_proj_weight;
    float *c_proj_bias;
    float *d_fc_weight;
    float *d_fc_bias;
    float *d_proj_weight;
    float *d_proj_bias;
    float *output;
    float *doutput;
};

struct LayerNorm
{
    float *input;
    float *ln1_weight;
    float *ln1_bias;
    float *cache_mean;
    float *cache_var;
    float *dln1_weight;
    float *dln1_bias;
    float *output;
    float *doutput;
};

struct DecoderLayer
{
    float *input;
    struct AttentionBlock *causal_attention;
    struct MLP *mlp;
    struct LayerNorm *ln1;
    struct LayerNorm *ln2;
    float *output;
    float *doutput;
};
struct FinalLayer
{
    float *final_weight;
    float *final_bias;
    float *dfinal_weight;
    float *dfinal_bias;
    float *output;
    float *doutput;
};

struct gpt
{
    struct Embedding *embedding;
    struct DecoderLayer *decoder_layer[8];
    struct LayerNorm *ln;
    struct FinalLayer *final_layer;
};

struct DecoderCache{
    float *mean;
    float *var;

};

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

void layernorm(const float *inp, struct LayerNorm ln, float *out, float *mean, float *std,int B, int T, int C, int train)
{
    float epsilon = 1e-5;
    if(train == 1){
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
                v += (1/C)*(inp[i * C + j] - m) * (inp[i * C + j] - m);
            }
            std[i] = sqrt(v + epsilon);
            float denom = std[i];
            ln.cache_mean[i] = 0.9f * ln.cache_mean[i] + 0.1f * m;
            ln.cache_var[i] = 0.9f * ln.cache_var[i] + 0.1f * denom;
            for (int j = 0; j < C; j++)
            {
                out[i * C + j] = ln.ln1_weight[j] * (inp[i * C + j] - m) / denom + ln.ln1_bias[j];
            }
        }

    }
    else{
        for(int i = 0; i < B * T; i++){
            for(int j = 0; j < C; j++){
                out[i * C + j] = ln.ln1_weight[j] * (inp[i*C + j] - ln.cache_mean[i])/ln.cache_var[i] + ln.ln1_bias[j];
            }
        }
    }
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

// manually derived the eqn for backprop
// B, T, C
// mean = sum(x, dim = -1)/c
// var = (x - mean)^2
// std = sqrt(var + epsilon)
// din = (dout * ln_weight)/std - sum(dout * ln_weight, dim = -1)/c*std - (x - mean) * sum(dout * ln_weight * (x - mean), dim = -1)/c*std^3
// splitting it into 3 parts
// var1 = (dout * ln_weight)/std ---> over B * T
// var2 = sum(dout * ln_weight dim = -1)/c*std ---> over C
// var3 = (x - mean) * sum(dout * ln_weight * (x - mean), dim = -1)/c*std^3 ---> over C
void layernorm_backprop(float *output, const float *input, const float *dout, float *din, struct LayerNorm ln, const float *mean, const float *std, int B, int T, int C) {
    for (int i = 0; i < B * T; i++) {
        float var2 = 0.0f;
        float var3 = 0.0f;
        
        // Calculate var2 and var3
        for (int j = 0; j < C; j++) {
            var2 += dout[i * C + j] * ln.ln1_weight[j];
            var3 += (input[i * C + j] - mean[i]) * dout[i * C + j] * ln.ln1_weight[j];
        }
        
        var2 = var2 / (C * std[i]);
        var3 = var3 / (C * std[i] * std[i] * std[i]);
        
        // Calculate gradients
        for (int j = 0; j < C; j++) {
            float norm = (input[i * C + j] - mean[i]) / std[i];
            float var1 = dout[i * C + j] * ln.ln1_weight[j] / std[i];
            
            din[i * C + j] = var1 - var2 - var3 * (input[i * C + j] - mean[i]);
            ln.dln1_weight[j] += dout[i * C + j] * norm;
            ln.dln1_bias[j] += dout[i * C + j];
        }
    }
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
void *safe_malloc(size_t size) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    return ptr;
}

void *safe_calloc(size_t num, size_t size) {
    void *ptr = calloc(num, size);
    if (ptr == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    return ptr;
}
// generating random numbers from normal distribution using Box Muller transform
float randn(float mean, float stddev) {
    static int use_last = 0;
    static float y2;
    float x1, x2, w, y1;

    if (use_last) {
        y1 = y2;
        use_last = 0;
    } else {
        do {
            x1 = 2.0 * ((float) rand() / RAND_MAX) - 1.0;
            x2 = 2.0 * ((float) rand() / RAND_MAX) - 1.0;
            w = x1 * x1 + x2 * x2;
        } while (w >= 1.0 || w == 0.0);

        w = sqrt((-2.0 * log(w)) / w);
        y1 = x1 * w;
        y2 = x2 * w;
        use_last = 1;
    }

    return (mean + y1 * stddev);
}

// Xavier normal initialization
void xavier_normal_init(float *tensor, int fan_in, int fan_out, int flag) {
    float stddev = sqrt(2.0 / (fan_in + fan_out));
    float x = 1 / sqrt(2.0*config.n_layers); // Adjusted handling for residual connections variance
    for (int i = 0; i < fan_in * fan_out; i++) {
        if (flag == 1) {
            tensor[i] = randn(0, stddev * x);
        } else {
            tensor[i] = randn(0, stddev);
        }
    }
}

void initialize_embedding(struct Embedding *emb, struct Config config) {
    emb->input = safe_calloc(config.batch_size * config.seq_len, sizeof(float));
    emb->pos_emb = safe_calloc(config.seq_len * config.d_model, sizeof(float));
    emb->inp_emb = safe_calloc(config.vocab_size * config.d_model, sizeof(float));
    emb->dinp_emb = safe_calloc(config.vocab_size * config.d_model, sizeof(float));
    emb->dpos_emb = safe_calloc(config.seq_len * config.d_model, sizeof(float));
    emb->output = safe_calloc(config.batch_size * config.seq_len * config.d_model, sizeof(float));

    xavier_normal_init(emb->inp_emb, config.vocab_size, config.d_model, 0);
    xavier_normal_init(emb->pos_emb, config.seq_len, config.d_model, 0);
}

void initialize_attention(struct AttentionBlock *attn, struct Config config) {
    attn->input = safe_calloc(config.batch_size * config.seq_len * config.d_model, sizeof(float));
    attn->attn_weight = safe_calloc(config.d_model * 3 * config.d_model, sizeof(float));
    attn->attn_bias = safe_calloc(3 * config.d_model, sizeof(float));
    attn->proj_weight = safe_calloc(config.d_model * config.d_model, sizeof(float));
    attn->proj_bias = safe_calloc(config.d_model, sizeof(float));
    attn->dattn_weight = safe_calloc(config.d_model * 3 * config.d_model, sizeof(float));
    attn->dattn_bias = safe_calloc(3 * config.d_model, sizeof(float));
    attn->dproj_weight = safe_calloc(config.d_model * config.d_model, sizeof(float));
    attn->dproj_bias = safe_calloc(config.d_model, sizeof(float));
    attn->output = safe_calloc(config.batch_size * config.seq_len * config.d_model, sizeof(float));

    xavier_normal_init(attn->attn_weight, config.d_model, 3 * config.d_model, 1);
    xavier_normal_init(attn->proj_weight, config.d_model, config.d_model, 1);
    xavier_normal_init(attn->attn_bias, 1, 3 * config.d_model, 0);
    xavier_normal_init(attn->proj_bias, 1, config.d_model, 0);
}

void initialize_mlp(struct MLP *mlp, struct Config config) {
    mlp->input = safe_calloc(config.batch_size * config.seq_len * config.d_model, sizeof(float));
    mlp->c_fc_weight = safe_calloc(config.d_model * 4 * config.d_model, sizeof(float));
    mlp->c_fc_bias = safe_calloc(4 * config.d_model, sizeof(float));
    mlp->c_proj_weight = safe_calloc(4 * config.d_model * config.d_model, sizeof(float));
    mlp->c_proj_bias = safe_calloc(config.d_model, sizeof(float));
    mlp->d_fc_weight = safe_calloc(config.d_model * 4 * config.d_model, sizeof(float));
    mlp->d_fc_bias = safe_calloc(4 * config.d_model, sizeof(float));
    mlp->d_proj_weight = safe_calloc(4 * config.d_model * config.d_model, sizeof(float));
    mlp->d_proj_bias = safe_calloc(config.d_model, sizeof(float));
    mlp->output = safe_calloc(config.batch_size * config.seq_len * config.d_model, sizeof(float));

    xavier_normal_init(mlp->c_fc_weight, config.d_model, 4 * config.d_model, 1);
    xavier_normal_init(mlp->c_proj_weight, 4 * config.d_model, config.d_model, 1);
    xavier_normal_init(mlp->c_fc_bias, 1, 4 * config.d_model, 0);
    xavier_normal_init(mlp->c_proj_bias, 1, config.d_model, 0);
}

void initialize_layernorm(struct LayerNorm *ln, struct Config config) {
    ln->input = safe_calloc(config.batch_size * config.seq_len * config.d_model, sizeof(float));
    ln->ln1_weight = safe_calloc(config.d_model, sizeof(float));
    ln->ln1_bias = safe_calloc(config.d_model, sizeof(float));
    ln->dln1_weight = safe_calloc(config.d_model, sizeof(float));
    ln->dln1_bias = safe_calloc(config.d_model, sizeof(float));
    ln->output = safe_calloc(config.batch_size * config.seq_len * config.d_model, sizeof(float));

    for (int i = 0; i < config.d_model; i++) {
        ln->ln1_weight[i] = 1.0f;
    }
}

void initialize_decoder_layer(struct DecoderLayer *layer, struct Config config) {
    layer->input = safe_calloc(config.batch_size * config.seq_len * config.d_model, sizeof(float));
    layer->causal_attention = safe_malloc(sizeof(struct AttentionBlock));
    layer->mlp = safe_malloc(sizeof(struct MLP));
    layer->ln1 = safe_malloc(sizeof(struct LayerNorm));
    layer->ln2 = safe_malloc(sizeof(struct LayerNorm));
    layer->output = safe_calloc(config.batch_size * config.seq_len * config.d_model, sizeof(float));

    initialize_attention(layer->causal_attention, config);
    initialize_mlp(layer->mlp, config);
    initialize_layernorm(layer->ln1, config);
    initialize_layernorm(layer->ln2, config);
}

void initialize_final_layer(struct FinalLayer *final, struct Config config) {
    final->final_weight = safe_calloc(config.vocab_size * config.d_model, sizeof(float));
    final->final_bias = safe_calloc(config.vocab_size, sizeof(float));
    final->dfinal_weight = safe_calloc(config.vocab_size * config.d_model, sizeof(float));
    final->dfinal_bias = safe_calloc(config.vocab_size, sizeof(float));

    xavier_normal_init(final->final_weight, config.d_model, config.vocab_size, 0);
    xavier_normal_init(final->final_bias, 1, config.vocab_size, 0);
}

void initialize_gpt(struct gpt *model, struct Config config) {
    model->embedding = safe_malloc(sizeof(struct Embedding));
    initialize_embedding(model->embedding, config);

    for (int i = 0; i < config.n_layers; i++) {
        model->decoder_layer[i] = safe_malloc(sizeof(struct DecoderLayer));
        initialize_decoder_layer(model->decoder_layer[i], config);
    }

    model->ln = safe_malloc(sizeof(struct LayerNorm));
    initialize_layernorm(model->ln, config);

    model->final_layer = safe_malloc(sizeof(struct FinalLayer));
    initialize_final_layer(model->final_layer, config);
}


int main() {
    struct Config config = {
        .batch_size = 32,
        .seq_len = 1024,
        .d_model = 768,
        .num_heads = 12,
        .vocab_size = 50257,
        .n_layers = 8
    };

    struct gpt model;
    initialize_gpt(&model, config);

    printf("GPT model initialized successfully!\n");

    return 0;
}



