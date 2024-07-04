#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "helper.c"
#define SQRT_2_OVER_PI 0.7978845608028654
#define const1 0.5
#define const2 0.044715

struct Config
{
    int batch_size;
    int seq_len;
    int num_heads;
    int d_model;
    int vocab_size;
    int n_layers;
};

struct Optimizer {
    float lr;
    float beta1;
    float beta2;
    float epsilon;
    float *m;
    float *v;
    float *t;
};

void ResidualLayer(float *input1, float *input2, float *output, int size) {
    printf("hi from residual\n");
    for (int i = 0; i < size; i++) {
        output[i] = input1[i] + input2[i];
    }
}
void residual_backprop(float *dinput1, float *dinput2, float *doutput, int size) {
    for (int i = 0; i < size; i++) {
        dinput1[i] += doutput[i];
        dinput2[i] += doutput[i];
    }
}
// // Cosine learning rate scheduler
// float lr_scheduler(float lr, int step, int warmup_steps, int total_steps) {
//     if (step < warmup_steps) {
//         return lr * (step / warmup_steps);
//     }
//     return 0.5 * lr * (1 + cos((step - warmup_steps) / (total_steps - warmup_steps) * M_PI));
// }
// GELU(x)=0.5*x*(1+Tanh(sqrt(2/π) * (x+0.044715*x^3))) tanh approximation refered from https://arxiv.org/pdf/1606.08415.pdf
void GELU(float *input, float *output, int size){
    for (int i = 0; i < size; i++) {
        float x = input[i];
        output[i] = const1 * x * (1 + tanh(SQRT_2_OVER_PI * (x + const2 * x * x * x)));
    }
}

void gelu_backprop(float *input, float *dinput, float *doutput, int size) {

    for (int i = 0; i < size; i++) {
        float x = input[i];
        float tanh_x = tanh(SQRT_2_OVER_PI * (x + const2 * x * x * x));
        float cdf = 0.5 * (1 + tanh_x) + 0.5 * x * (1 - tanh_x * tanh_x) * SQRT_2_OVER_PI * (1 + 3 * const2 * x * x);
        dinput[i] = cdf * doutput[i];
    }
}

void matmul(const float *inp, const float *weight, float *bias, float *out, int m, int k, int n, int B)
{
    
    #pragma omp parallel for collapse(2)
    for (int batch = 0; batch < B; batch++)
    {
        for (int i = 0; i < m; i++)
        {
            int b_i = batch * m * n + i * n;
            for (int l = 0; l < k; l++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (l == 0)
                    {
                        out[b_i + j] = bias == NULL ? 0.0f : bias[j];
                    }
                    out[b_i + j] += inp[batch * m * k + i * k + l] * weight[l * n + j];
                }
            }
        }
    }
}
// out = inp @ weight + bias--> dinp = dout @ (weight).T
// dweight = (dinp).T @ dout, dbias = dout.sum()
// Transposing and matrix mul gets lil bit hairy, i tried on a napkin first
// bad naming of vars but dout is actually the gradient coming into the block not going 'out' of the block
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
            for (int x = 0; x < k; x++) {
                for (int l = 0; l < n; l++) {
                    dweight[x * n + l] += inp[i * m * k + j * k + x] * dout[i * m * n + j * n + l];
                }
            }
        }
    }
}

void embeddings_layer(const int *inp, const float *inp_emb, const float *pos_emb, float *out, int B, int T, int C)
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
void embedding_backprop(const float *doutput, const int *inp, float *dinp_emb, float *dpos_emb, int B, int T, int C)
{
    for (int i = 0; i < B; i++)
    {
        for (int j = 0; j < T; j++)
        {
            for (int k = 0; k < C; k++)
            {
                dinp_emb[inp[i * T + j] * C + k] += doutput[i * T * C + j * C + k];
                dpos_emb[j * C + k] += doutput[i * T * C + j * C + k];
            }
        }
    }
}
void layernorm(float *input, float *output, float *mean, struct LayerNorm *ln, float *std, int B, int T, int C, int train)
{
    float epsilon = 1e-5;
    if (train == 1) {
        float c = C;
        for (int i = 0; i < B * T; i++)
        {
            float sum = 0.0f;
            for (int j = 0; j < C; j++)
            {
                sum += input[i * C + j];
            }
            mean[i] = sum / c;
            float m = mean[i];
            float v = 0.0f;
            for (int j = 0; j < C; j++)
            {
                v += (1 / C) * (input[i * C + j] - m) * (input[i * C + j] - m);
            }
            std[i] = sqrt(v + epsilon);
            float denom = std[i];
            ln->cache_mean[i] = 0.9f * ln->cache_mean[i] + 0.1f * m;
            ln->cache_var[i] = 0.9f * ln->cache_var[i] + 0.1f * denom;
            for (int j = 0; j < C; j++)
            {
                output[i * C + j] = ln->ln1_weight[j] * (input[i * C + j] - m) / denom + ln->ln1_bias[j];
            }
        }

    }
    else {
        for (int i = 0; i < B * T; i++) {
            for (int j = 0; j < C; j++) {
                output[i * C + j] = ln->ln1_weight[j] * (input[i * C + j] - ln->cache_mean[i]) / ln->cache_var[i] + ln->ln1_bias[j];
            }
        }
    }
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
void layernorm_backprop(float *input, float *dinput, float *doutput, struct LayerNorm *ln, struct dLayerNorm *dln, const float *mean, const float *std, int B, int T, int C) {
    for (int i = 0; i < B * T; i++) {
        float var2 = 0.0f;
        float var3 = 0.0f;

        // Calculate var2 and var3
        for (int j = 0; j < C; j++) {
            var2 += doutput[i * C + j] * ln->ln1_weight[j];
            var3 += (input[i * C + j] - mean[i]) * doutput[i * C + j] * ln->ln1_weight[j];
        }

        var2 = var2 / (C * std[i]);
        var3 = var3 / (C * std[i] * std[i] * std[i]);

        // Calculate gradients
        for (int j = 0; j < C; j++) {
            float norm = (input[i * C + j] - mean[i]) / std[i];
            float var1 = doutput[i * C + j] * ln->ln1_weight[j] / std[i];

            dinput[i * C + j] = var1 - var2 - var3 * (input[i * C + j] - mean[i]);
            dln->dln1_weight[j] += doutput[i * C + j] * norm;
            dln->dln1_bias[j] += doutput[i * C + j];
        }
    }
}

void multilayer_perceptron(float *input, float *output1, float *output2, float *gelu_out1, float *gelu_out2, struct MLP *mlp, int B, int T, int C)
{
    // b, t, c = b, t, 4c
    matmul(input, mlp->c_fc_weight, mlp->c_fc_bias, output1, T, C, 4 * C, B);// output of first layer is with gelu input
    GELU(output1, gelu_out1, B * T * 4 * C);
    //b, t, 4c = b, t, c
    matmul(gelu_out1, mlp->c_proj_weight, mlp->c_proj_bias, output2, T, 4 * C, C, B);
    GELU(output2, gelu_out2, B * T * C);
}

void mlp_backprop(float *input, float *dinput, struct ip_op *io, struct MLP *mlp, struct dMlp *dmlp, int B, int T, int C) {
    gelu_backprop(io->mlp_l2_out, io->dmlp_l2_out, io->dgelu_out2, B * T * C);
    matmul_backprop(io->gelu_out1, mlp->c_proj_weight, io->dmlp_l2_out, io->dgelu_out1, dmlp->d_proj_weight, dmlp->d_proj_bias, T, 4 * C, C, B);
    gelu_backprop(io->mlp_l1_out, io->dmlp_l1_out, io->dgelu_out1, B * T * 4 * C);
    matmul_backprop(input, mlp->c_fc_weight, io->dmlp_l1_out, dinput, dmlp->d_fc_weight, dmlp->d_fc_bias, T, C, 4 * C, B);
}


void CausalAttention(float *input, float *output, struct AttentionBlock *attn, struct ip_op *io, int B, int T, int C, int num_heads)
{
    assert(C % num_heads == 0);
    int head_size = C / num_heads;
    int N = num_heads;
    // --------- Stage 1-----------
    // inp @ qkv = B, T, C @ C, 3C = B, T, 3C
    // --------- Stage 2-----------
    // Q @ K.T = B, N, T, H @ B, N, H, T = (B, N, T, T)/sqrt(H)
    // self attention = apply mask and softmax
    // self attention  @ V = B, N, T, T @ B, N, T, H = B, N, T, H = attention score
    // --------- Stage 3-----------
    // attention score transpose(1,2) = B, T, N, H
    // B, T, N, H @ C, C + 1,C = B, T, C--->output

    matmul(input, attn->attn_weight, attn->attn_bias, io->qkv, T, C, 3 * C, B);
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < B; i++) {
        for (int n = 0; n < N; n++) {
            for (int j = 0; j < T; j++) {

                float *query = io->qkv + i * T * 3 * C + j * 3 * C + n * head_size;
                float maxi = -INFINITY;
                // stage 2--> Q @ K
                for (int k = 0; k < T; k++) {

                    float *key = io->qkv + i * T * 3 * C + k * 3 * C + n * head_size + C;
                    float score = 0.0f;
                    for (int h = 0; h < head_size; h++) {
                        score += (query[h] * key[h]) / (sqrt(head_size));
                    }
                    io->attn_scores[i * N * T * T + n * T * T + j * T + k] = score;
                    maxi = maxi > score ? maxi : score;
                }
                // we have calculated scores for a single row in the T x T self att matrix, now we apply softmax and matmul with V
                //softmax(x) = exp(x - max)/sum(exp(x - max)), x-max for numerical stability
                float expsum = 0.0f;
                for (int k = 0; k < T; k++) {
                    if (k <= j) {
                        expsum += exp(io->attn_scores[i * N * T * T + n * T * T + j * T + k] - maxi);// lower triangular matrix
                    }
                    else {
                        io->attn_scores[i * N * T * T + n * T * T + j * T + k] = -INFINITY;
                    }
                }
                for (int k = 0; k < T; k++) {
                    io->attn_scores[i * N * T * T + n * T * T + j * T + k] = exp(io->attn_scores[i * N * T * T + n * T * T + j * T + k] - maxi) / expsum;
                }
                // stage 3 --> attn_scores @ V
                for (int k = 0; k < T; k++) {
                    float *value = io->qkv + i * T * 3 * C + k * 3 * C + n * head_size + 2 * C;
                    for (int h = 0; h < head_size; h++) {
                        io->causal_attention[i * T * C + j * C + n * head_size + h] += io->attn_scores[i * N * T * T + n * T * T + j * T + k] * value[h];
                    }
                }
            }
        }
    }
    // output projection layer
    matmul(io->causal_attention, attn->proj_weight, attn->proj_bias, output, T, C, C, B);
    // one of the most annoying and fun things to code
}

void causal_attention_backprop(struct AttentionBlock *attn, struct dAttentionBlock *dAtt, struct ip_op *io, int B, int T, int C, int num_heads) {

    float *dattn = (float *)calloc(B * T * C, sizeof(float));
    int N = num_heads;
    int H = C / N;
    // stage 3
    matmul_backprop(io->causal_attention, attn->proj_weight, io->datt_out, dattn, dAtt->dproj_weight, dAtt->dproj_bias, T, C, C, B);
    //stage 2
    float *dqkv = (float *)calloc(B * T * 3 * C, sizeof(float));
    float *dscores = (float *)calloc(B * N * T * T, sizeof(float));
    #pragma omp parallel for collapse(3)
    
    for(int i = 0; i < B; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < T; k++){
                int idx1 = i * T * 3 * C + k * 3 * C + j * H;// indexing qkv
                int idx2 = i * T * T * N + j * T * T + k * T;// indexing attn_scores
                int idx3 = i * T * C + k * C + j * H;// indexing attn
                for(int l = 0; l < T; l++){
                    for(int h = 0; h < H; h++){
                        // attn_scores--> T, T  ; value --> T, H
                        dqkv[idx1 + 2*C + h] += io->attn_scores[idx2 + l] * dattn[h];
                        dscores[idx2 + l] += io->qkv[idx1 + 2*C + h] * dattn[h];
                    }
                }
                // now for softmax's turn, in forward pass we filled upper triangular matrix with -inf and simultaneously calculated exp sum finally applying exp/expsum
                // wonderful blog https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
                // derivative = softmax(x) * (1 - softmax(x)) if input idx(x) == ouput idx(y) else -softmax(x) * softmax(y)
                float *dattn_raw = (float *)calloc(T, sizeof(float));
                for(int l = 0; l < T; l++){
                    float temp = io->attn_scores[idx2 + l];
                    float der = dscores[idx2 + l];
                    if(l == k){
                        dattn_raw[l] = (temp * (1 - temp))*der;
                    }
                    else{
                        dattn_raw[l] = (-temp * io->attn_scores[idx2 + k])*der;
                    }
                }
                // now attn_raw = (q @ k)/sqrt(H), attn_raw--> T, T; q--> T, H; k--> T, H
                // dq = dattn_raw @ K 
                // dk = dattn_raw @ Q
                for(int l = 0; l < T; l++){
                    for(int h = 0; h < H; h++){
                        dqkv[idx1 + h] = io->qkv[idx1 + C + h] * dattn_raw[l];
                        dqkv[idx1 + C + h] = io->qkv[idx1 + h] * dattn_raw[l];
                    }
                }
            }
        }
    }
    matmul_backprop(io->ln_out1, attn->attn_weight, dqkv, io->dln_out1, dAtt->dattn_weight, dAtt->dattn_bias, T, 3 * C, C, B);
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


void loss_backward(float *logits, int *targets, int B, int T, int C)
{
    float *dlogits = (float *)calloc(B * T * C, sizeof(float));
    Softmax(logits, dlogits, B, T, C);
    for (int i = 0; i < B * T; i++)
    {
        for(int j = 0; j < C; j++){
            dlogits[i * C + j] -= targets[i] == j ? 1.0f : 0.0f;
            dlogits[i * C + j] /= B * T;
        }
    }
}

void gpt_forward(struct Config config, struct gpt *model, struct ip_op *io[], struct foo *f){
    int B = config.batch_size;
    int T = config.seq_len;
    int C = config.d_model;
    int N = config.num_heads;
    int n = config.n_layers;
    // embeddings layer
    embeddings_layer(f->input, model->embedding->inp_emb, model->embedding->pos_emb, f->emb_output, B, T, C);
    // decoder layers
    for(int i = 0; i < n; i++){
        if(i==0){layernorm(f->emb_output, io[i]->ln_out1, io[i]->mean1, model->decoder_layer[i]->ln1, io[i]->std1, B, T, C, 1);}
        else{layernorm(io[i-1]->res_out2, io[i]->ln_out1, io[i]->mean1, model->decoder_layer[i]->ln1, io[i]->std1, B, T, C, 1);}
        CausalAttention(io[i]->ln_out1, io[i]->att_out, model->decoder_layer[i]->causal_attention, io[i], B, T, C, N);
        if(i==0){ResidualLayer(io[i]->att_out, f->emb_output, io[i]->res_out1, B * T * C);}
        else{ResidualLayer(io[i]->att_out, io[i-1]->res_out2, io[i]->res_out1, B * T * C);}
        layernorm(io[i]->res_out1, io[i]->ln_out2, io[i]->mean2, model->decoder_layer[i]->ln2, io[i]->std2, B, T, C, 1);
        multilayer_perceptron(io[i]->ln_out2, io[i]->mlp_l1_out, io[i]->mlp_l2_out, io[i]->gelu_out1, io[i]->gelu_out2, model->decoder_layer[i]->mlp, B, T, C);
        ResidualLayer(io[i]->mlp_l2_out, io[i]->ln_out2, io[i]->res_out2, B * T * C);
    }
    // final linear layer
    layernorm(io[n - 1]->res_out2, f->fln_output, f->fln_mean, model->lnf, f->fln_std, B, T, C, 1);
    matmul(f->fln_output, model->final_layer->final_weight, model->final_layer->final_bias, f->fl_output, T, C, config.vocab_size, B);
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

void xavier_normal_init(float *tensor, int fan_in, int fan_out, int flag) {
    float stddev = sqrt(2.0 / (fan_in + fan_out));
    float x = 1 / sqrt(2.0 * 8); // Adjusted handling for residual connections variance
    for (int i = 0; i < fan_in * fan_out; i++) {
        if (flag == 1) {
            tensor[i] = randn(0, stddev * x);
        } else {
            tensor[i] = randn(0, stddev);
        }
    }
}


int main() {
    struct Config config = {16, 1024, 8, 768, 50304, 8};
    int B = config.batch_size, T = config.seq_len, C = config.d_model, V = config.vocab_size, N = config.num_heads, NUM_LAYERS = config.n_layers;
    struct gpt *model = (struct gpt *)malloc(sizeof(struct gpt));
    struct dGpt *dmodel = (struct dGpt *)malloc(sizeof(struct dGpt));
    struct ip_op **io = (struct ip_op **)malloc(NUM_LAYERS * sizeof(struct ip_op *));
    struct foo *f = (struct foo *)malloc(sizeof(struct foo));
    map_gpt(model, "gpt.bin", B, T, C, V);
    map_dgpt(dmodel, "dgpt_model.dat", T, C, V);
    map_ip_op_array(io, "ip_op.bin", B, T, C, NUM_LAYERS, N);
    map_foo(f, "foo.bin", B, T, C);
    clock_t start_time = clock();
    gpt_forward(config, model, io, f);
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) * 1000.0 / CLOCKS_PER_SEC;
    printf("Execution time: %.2f milliseconds\n", elapsed_time);
    int i = 6;
    

    printf("%f", (f->emb_output[16*1024*768]));
    return 0;
}
