#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include "mapfile.c"

struct ip_op{

    float *dec_ip;//b, t, c
    float *ln_out1;//b, t, c
    float *mean1;//b, t
    float *std1;//b, t
    float *att_out;//b, t, c
    float *causal_attention;//b, t, c
    float *attn_scores;//b, n, t, t
    float *qkv;// b, t, 3c
    float *res_out1;//b, t, c
    float *ln_out2;//b, t, c
    float *mean2;//b, t
    float *std2;//b, t
    float *mlp_l1_out;//b, t, 4c
    float *gelu_out1;//b, t, 4c
    float *mlp_l2_out;//b, t, c
    float *gelu_out2;//b, t, c
    float *res_out2;//b, t, c

    // Backprop

    float *dln_out1;//b, t, c
    float *datt_out;//b, t, c
    float *dres_out1;//b, t, c
    float *dln_out2;//b, t, c
    float *dmlp_l1_out;//b, t, 4c
    float *dgelu_out1;//b, t, 4c
    float *dmlp_l2_out;//b, t, c
    float *dgelu_out2;//b, t, c
    float *dres_out2;//b, t, c

};

// contains remaining values of the gpt apart from decoder eg: embedding, final layer

struct foo{
    int *input;// b, t
    float *emb_output;// b, t, c
    float *fln_output;// b, t, c
    float *fln_mean;// b, t
    float *fln_std;// b, t
    float *fl_output;// b, t, c
    // Backprop
    float *demb_output;// b, t, c
    float *dfln_output;// b, t, c
    float *dfl_output;// b, t, c
};

struct Embedding{
    float *pos_emb;//t, c
    float *inp_emb;//v, c
};

struct dEmbedding{
    float *dinp_emb;//v, c
    float *dpos_emb;//t, c
    float *doutput;//b, t, c
};

struct AttentionBlock{
    float *attn_weight;//c, 3c
    float *attn_bias;//3c
    float *proj_weight;//c, c
    float *proj_bias;//c
};


struct dAttentionBlock{
    float *dattn_weight;//c, 3c
    float *dattn_bias;//3c
    float *dproj_weight;//c, c
    float *dproj_bias;//c
};

struct MLP{
    float *c_fc_weight;//c, 4c
    float *c_fc_bias;//4c
    float *c_proj_weight;//4c, c
    float *c_proj_bias;//c
};

struct dMlp{
    float *d_fc_weight;//c, 4c
    float *d_fc_bias;//4c
    float *d_proj_weight;//4c, c
    float *d_proj_bias;//c
};

struct LayerNorm{
    float *ln1_weight;//c
    float *ln1_bias;//c
    float *cache_mean;//b, t
    float *cache_var;//b, t
};

struct dLayerNorm{
    float *dln1_weight;//c
    float *dln1_bias;//c
};

struct DecoderLayer{
    struct AttentionBlock *causal_attention;
    struct MLP *mlp;
    struct LayerNorm *ln1;
    struct LayerNorm *ln2;
};

struct dDecoderLayer{
    struct dAttentionBlock *dcausal_attention;
    struct dMlp *dmlp;
    struct dLayerNorm *dln1;
    struct dLayerNorm *dln2;

};

struct FinalLayer{
    float *final_weight;//c, v
    float *final_bias;//v
};

struct dFinalLayer{
    float *dfinal_weight;//c, v
    float *dfinal_bias;//v
};

struct gpt{
    struct Embedding *embedding;
    struct DecoderLayer *decoder_layer[8];
    struct LayerNorm *lnf;
    struct FinalLayer *final_layer;
};

struct dGpt{
    struct dEmbedding *dembedding;
    struct dDecoderLayer *ddecoder_layer[8];
    struct dLayerNorm *dlnf;
    struct dFinalLayer *dfinal_layer;
};

// allocating 3 huge chunks of memory and 1 small one using memory pooling:

// 1. for forward pass --> gpt containing weights and biases

// 2. for backward pass --> dgpt containing gradients

// 3. ip_op containing intermediate inputs, outputs, etc of each decoder layer

// 4. foo containing remaining intermediate inputs, outputs, etc of embedding and final layer
size_t calculate_embedding_size(int t, int c, int v) {
    return (t * c + v * c) * sizeof(float);
}

size_t calculate_attentionblock_size(int c) {
    return (c * 3 * c + 3 * c + c * c + c) * sizeof(float);
}

size_t calculate_mlp_size(int c) {
    return (c * 4 * c + 4 * c + 4 * c * c + c) * sizeof(float);
}

size_t calculate_layernorm_size(int c, int b, int t) {
    return (2 * c + 2 * 2 * b * t) * sizeof(float);
}

size_t calculate_final_layer_size(int c, int v) {
    return (c * v + v) * sizeof(float);
}

size_t calculate_total_gpt_size(int b, int t, int c, int v) {
    size_t total_size = 0;

    // Embedding size
    total_size += calculate_embedding_size(t, c, v);

    // Decoder layers size
    for (int i = 0; i < 8; i++) {
        total_size += calculate_attentionblock_size(c);
        total_size += calculate_mlp_size(c);
        total_size += calculate_layernorm_size(c, b , t) * 2; // ln1 and ln2
    }

    // Final layer norm size
    total_size += calculate_layernorm_size(c, b , t);

    // Final layer size
    total_size += calculate_final_layer_size(c, v);

    return total_size;
}
size_t calculate_dembedding_size(int t, int c, int v) {
    return (v * c + t * c + t * c) * sizeof(float);
}

size_t calculate_dattentionblock_size(int c) {
    return (c * 3 * c + 3 * c + c * c + c) * sizeof(float);
}

size_t calculate_dmlp_size(int c) {
    return (c * 4 * c + 4 * c + 4 * c * c + c) * sizeof(float);
}

size_t calculate_dlayernorm_size(int c) {
    return (2 * c) * sizeof(float);
}

size_t calculate_total_dgpt_size(int t, int c, int v) {
    size_t total_size = 0;

    // dEmbedding size
    total_size += calculate_dembedding_size(t, c, v);

    // dDecoder layers size
    for (int i = 0; i < 8; i++) {
        total_size += calculate_dattentionblock_size(c);
        total_size += calculate_dmlp_size(c);
        total_size += calculate_dlayernorm_size(c) * 2; // dln1 and dln2
    }

    // dFinal layer norm size
    total_size += calculate_dlayernorm_size(c);

    // dFinal layer size
    total_size += calculate_final_layer_size(c, v); // Assuming dFinal layer is the same size as Final layer

    return total_size;
}

void map_gpt(struct gpt* model, const char* filename, int b, int t, int c, int v) {
    size_t total_size = calculate_total_gpt_size(b, t, c, v);
    float* map = mmap_file(filename, total_size);
    if (!map) {
        perror("mmap_file");
        return;
    }

    float* current_ptr = map;

    // Map embedding
    model->embedding = (struct Embedding*)malloc(sizeof(struct Embedding));
    model->embedding->inp_emb = current_ptr;
    current_ptr += v * c;
    model->embedding->pos_emb = current_ptr;
    current_ptr += t * c;
    

    // Map decoder layers
    for (int i = 0; i < 8; i++) {
        model->decoder_layer[i] = (struct DecoderLayer*)malloc(sizeof(struct DecoderLayer));

        model->decoder_layer[i]->causal_attention = (struct AttentionBlock*)malloc(sizeof(struct AttentionBlock));
        model->decoder_layer[i]->causal_attention->attn_weight = current_ptr;
        current_ptr += c * 3 * c;
        model->decoder_layer[i]->causal_attention->attn_bias = current_ptr;
        current_ptr += 3 * c;
        model->decoder_layer[i]->causal_attention->proj_weight = current_ptr;
        current_ptr += c * c;
        model->decoder_layer[i]->causal_attention->proj_bias = current_ptr;
        current_ptr += c;

        model->decoder_layer[i]->mlp = (struct MLP*)malloc(sizeof(struct MLP));
        model->decoder_layer[i]->mlp->c_fc_weight = current_ptr;
        current_ptr += c * 4 * c;
        model->decoder_layer[i]->mlp->c_fc_bias = current_ptr;
        current_ptr += 4 * c;
        model->decoder_layer[i]->mlp->c_proj_weight = current_ptr;
        current_ptr += 4 * c * c;
        model->decoder_layer[i]->mlp->c_proj_bias = current_ptr;
        current_ptr += c;

        model->decoder_layer[i]->ln1 = (struct LayerNorm*)malloc(sizeof(struct LayerNorm));
        model->decoder_layer[i]->ln1->ln1_weight = current_ptr;
        current_ptr += c;
        model->decoder_layer[i]->ln1->ln1_bias = current_ptr;
        current_ptr += c;
        model->decoder_layer[i]->ln1->cache_mean = current_ptr;
        current_ptr += b * t;
        model->decoder_layer[i]->ln1->cache_var = current_ptr;
        current_ptr += b * t;

        model->decoder_layer[i]->ln2 = (struct LayerNorm*)malloc(sizeof(struct LayerNorm));
        model->decoder_layer[i]->ln2->ln1_weight = current_ptr;
        current_ptr += c;
        model->decoder_layer[i]->ln2->ln1_bias = current_ptr;
        current_ptr += c;
        model->decoder_layer[i]->ln2->cache_mean = current_ptr;
        current_ptr += b * t;
        model->decoder_layer[i]->ln2->cache_var = current_ptr;
        current_ptr += b * t;
    }

    // Map final layer norm
    model->lnf = (struct LayerNorm*)malloc(sizeof(struct LayerNorm));
    model->lnf->ln1_weight = current_ptr;
    current_ptr += c;
    model->lnf->ln1_bias = current_ptr;
    current_ptr += c;
    model->lnf->cache_mean = current_ptr;
    current_ptr += b*t;
    model->lnf->cache_var = current_ptr;
    current_ptr += b*t;

    // Map final layer
    model->final_layer = (struct FinalLayer*)malloc(sizeof(struct FinalLayer));
    model->final_layer->final_weight = current_ptr;
    current_ptr += c * v;
    model->final_layer->final_bias = current_ptr;
    current_ptr += v;
}

void unmap_gpt(struct gpt* model, const char* filename, int b, int t, int c, int v) {
    size_t total_size = calculate_total_gpt_size(b, t, c, v);
    float* map = model->embedding->pos_emb; // Assuming pos_emb points to the start of the mapped file

    // Free embedding
    free(model->embedding);

    // Free decoder layers
    for (int i = 0; i < 8; i++) {
        free(model->decoder_layer[i]->causal_attention);
        free(model->decoder_layer[i]->mlp);
        free(model->decoder_layer[i]->ln1);
        free(model->decoder_layer[i]->ln2);
        free(model->decoder_layer[i]);
    }

    // Free final layer norm
    free(model->lnf);

    // Free final layer
    free(model->final_layer);

    // Unmap file
    unmap_file(map, total_size);
}
void map_dgpt(struct dGpt* model, const char* filename, int t, int c, int v) {
    size_t total_size = calculate_total_dgpt_size(t, c, v);
    float* map = mmap_file(filename, total_size);
    if (!map) {
        perror("mmap_file");
        return;
    }

    float* current_ptr = map;

    // Map dEmbedding
    model->dembedding = (struct dEmbedding*)malloc(sizeof(struct dEmbedding));
    model->dembedding->dinp_emb = current_ptr;
    current_ptr += v * c;
    model->dembedding->dpos_emb = current_ptr;
    current_ptr += t * c;
    model->dembedding->doutput = current_ptr;
    current_ptr += t * c;

    // Map dDecoder layers
    for (int i = 0; i < 8; i++) {
        model->ddecoder_layer[i] = (struct dDecoderLayer*)malloc(sizeof(struct dDecoderLayer));

        model->ddecoder_layer[i]->dcausal_attention = (struct dAttentionBlock*)malloc(sizeof(struct dAttentionBlock));
        model->ddecoder_layer[i]->dcausal_attention->dattn_weight = current_ptr;
        current_ptr += c * 3 * c;
        model->ddecoder_layer[i]->dcausal_attention->dattn_bias = current_ptr;
        current_ptr += 3 * c;
        model->ddecoder_layer[i]->dcausal_attention->dproj_weight = current_ptr;
        current_ptr += c * c;
        model->ddecoder_layer[i]->dcausal_attention->dproj_bias = current_ptr;
        current_ptr += c;

        model->ddecoder_layer[i]->dmlp = (struct dMlp*)malloc(sizeof(struct dMlp));
        model->ddecoder_layer[i]->dmlp->d_fc_weight = current_ptr;
        current_ptr += c * 4 * c;
        model->ddecoder_layer[i]->dmlp->d_fc_bias = current_ptr;
        current_ptr += 4 * c;
        model->ddecoder_layer[i]->dmlp->d_proj_weight = current_ptr;
        current_ptr += 4 * c * c;
        model->ddecoder_layer[i]->dmlp->d_proj_bias = current_ptr;
        current_ptr += c;

        model->ddecoder_layer[i]->dln1 = (struct dLayerNorm*)malloc(sizeof(struct dLayerNorm));
        model->ddecoder_layer[i]->dln1->dln1_weight = current_ptr;
        current_ptr += c;
        model->ddecoder_layer[i]->dln1->dln1_bias = current_ptr;
        current_ptr += c;

        model->ddecoder_layer[i]->dln2 = (struct dLayerNorm*)malloc(sizeof(struct dLayerNorm));
        model->ddecoder_layer[i]->dln2->dln1_weight = current_ptr;
        current_ptr += c;
        model->ddecoder_layer[i]->dln2->dln1_bias = current_ptr;
        current_ptr += c;

    }

    // Map dFinal layer norm
    model->dlnf = (struct dLayerNorm*)malloc(sizeof(struct dLayerNorm));
    model->dlnf->dln1_weight = current_ptr;
    current_ptr += c;
    model->dlnf->dln1_bias = current_ptr;
    current_ptr += c;

    // Map dFinal layer
    model->dfinal_layer = (struct dFinalLayer*)malloc(sizeof(struct dFinalLayer));
    model->dfinal_layer->dfinal_weight = current_ptr;
    current_ptr += c * v;
    model->dfinal_layer->dfinal_bias = current_ptr;
    current_ptr += v;
}

void unmap_dgpt(struct dGpt* model, const char* filename, int t, int c, int v) {
    size_t total_size = calculate_total_dgpt_size(t, c, v);
    float* map = model->dembedding->dinp_emb; // Assuming dinp_emb points to the start of the mapped file

    // Free dEmbedding
    free(model->dembedding);

    // Free dDecoder layers
    for (int i = 0; i < 8; i++) {
        free(model->ddecoder_layer[i]->dcausal_attention);
        free(model->ddecoder_layer[i]->dmlp);
        free(model->ddecoder_layer[i]->dln1);
        free(model->ddecoder_layer[i]->dln2);
        free(model->ddecoder_layer[i]);
    }

    // Free dFinal layer norm
    free(model->dlnf);

    // Free dFinal layer
    free(model->dfinal_layer);

    // Unmap file
    unmap_file(map, total_size);
}

void map_ip_op_array(struct ip_op* model[], const char* filename, int b, int t, int c, int n, int N) {
    size_t total_size = n * (
        sizeof(float) * (b * t * c +  // dec_ip
                         b * t * c +  // ln_out1
                         b * t +      // mean1
                         b * t +      // std1
                         b * t * c +  // att_out
                         b * t * c +  // causal_attention
                         b * N * t * t +  // attn_scores
                         b * t * 3 * c +  // qkv
                         b * t * c +  // res_out1
                         b * t * c +  // ln_out2
                         b * t +      // mean2
                         b * t +      // std2
                         b * t * 4 * c +  // mlp_l1_out
                         b * t * 4 * c +  // gelu_out1
                         b * t * c +  // mlp_l2_out
                         b * t * c +  // gelu_out2
                         b * t * c +  // res_out2
                         b * t * c +  // dln_out1
                         b * t * c +  // datt_out
                         b * t * c +  // dres_out1
                         b * t * c +  // dln_out2
                         b * t * 4 * c +  // dmlp_l1_out
                         b * t * 4 * c +  // dgelu_out1
                         b * t * c +  // dmlp_l2_out
                         b * t * c +  // dgelu_out2
                         b * t * c     // dres_out2
                         )
    );  // Calculate total size to map

    float* map = mmap_file(filename, total_size);
    if (!map) {
        perror("mmap_file");
        return;
    }

    float* current_ptr = map;

    for (int i = 0; i < n; i++) {
        model[i] = (struct ip_op*)malloc(sizeof(struct ip_op));

        model[i]->dec_ip = current_ptr;
        current_ptr += b * t * c;

        model[i]->ln_out1 = current_ptr;
        current_ptr += b * t * c;

        model[i]->mean1 = current_ptr;
        current_ptr += b * t;

        model[i]->std1 = current_ptr;
        current_ptr += b * t;

        model[i]->att_out = current_ptr;
        current_ptr += b * t * c;

        model[i]->causal_attention = current_ptr;
        current_ptr += b * t * c;

        model[i]->attn_scores = current_ptr;
        current_ptr += b * n * t * t;

        model[i]->qkv = current_ptr;
        current_ptr += b * t * 3 * c;

        model[i]->res_out1 = current_ptr;
        current_ptr += b * t * c;

        model[i]->ln_out2 = current_ptr;
        current_ptr += b * t * c;

        model[i]->mean2 = current_ptr;
        current_ptr += b * t;

        model[i]->std2 = current_ptr;
        current_ptr += b * t;

        model[i]->mlp_l1_out = current_ptr;
        current_ptr += b * t * 4 * c;

        model[i]->gelu_out1 = current_ptr;
        current_ptr += b * t * 4 * c;

        model[i]->mlp_l2_out = current_ptr;
        current_ptr += b * t * c;

        model[i]->gelu_out2 = current_ptr;
        current_ptr += b * t * c;

        model[i]->res_out2 = current_ptr;
        current_ptr += b * t * c;

        // Backpropagation pointers
        model[i]->dln_out1 = current_ptr;
        current_ptr += b * t * c;

        model[i]->datt_out = current_ptr;
        current_ptr += b * t * c;

        model[i]->dres_out1 = current_ptr;
        current_ptr += b * t * c;

        model[i]->dln_out2 = current_ptr;
        current_ptr += b * t * c;

        model[i]->dmlp_l1_out = current_ptr;
        current_ptr += b * t * 4 * c;

        model[i]->dgelu_out1 = current_ptr;
        current_ptr += b * t * 4 * c;

        model[i]->dmlp_l2_out = current_ptr;
        current_ptr += b * t * c;

        model[i]->dgelu_out2 = current_ptr;
        current_ptr += b * t * c;

        model[i]->dres_out2 = current_ptr;
        current_ptr += b * t * c;
    }
}
void unmap_ip_op_array(struct ip_op* model[], int b, int t, int c, int n) {
    for (int i = 0; i < b * n; i++) {
        free(model[i]);
    }

    // Unmap file
    float* map = model[0]->dec_ip; // Assuming dec_ip points to the start of the mapped file
    size_t total_size = n * (
        sizeof(float) * (b * t * c +  // dec_ip
                         b * t * c +  // ln_out1
                         b * t +      // mean1
                         b * t +      // std1
                         b * t * c +  // att_out
                         b * t * c +  // causal_attention
                         b * n * t * t +  // attn_scores
                         b * t * 3 * c +  // qkv
                         b * t * c +  // res_out1
                         b * t * c +  // ln_out2
                         b * t +      // mean2
                         b * t +      // std2
                         b * t * 4 * c +  // mlp_l1_out
                         b * t * 4 * c +  // gelu_out1
                         b * t * c +  // mlp_l2_out
                         b * t * c +  // gelu_out2
                         b * t * c +  // res_out2
                         b * t * c +  // dln_out1
                         b * t * c +  // datt_out
                         b * t * c +  // dres_out1
                         b * t * c +  // dln_out2
                         b * t * 4 * c +  // dmlp_l1_out
                         b * t * 4 * c +  // dgelu_out1
                         b * t * c +  // dmlp_l2_out
                         b * t * c +  // dgelu_out2
                         b * t * c     // dres_out2
                         )
    );

    unmap_file(map, total_size);
}
void map_foo(struct foo* model, const char* filename, int b, int t, int c) {
    size_t total_size = b * (
        sizeof(int) * t +              // input
        sizeof(float) * t * c +        // emb_output
        sizeof(float) * t * c +        // fln_output
        sizeof(float) * t +            // fln_mean
        sizeof(float) * t +            // fln_std
        sizeof(float) * t * c +        // fl_output
        sizeof(float) * t * c +        // demb_output
        sizeof(float) * t * c +        // dfln_output
        sizeof(float) * t * c          // dfl_output
    );

    float* map = mmap_file(filename, total_size);
    if (!map) {
        perror("mmap_file");
        return;
    }

    float* current_ptr = map;

    model->input = (int*)current_ptr;
    current_ptr += b * t;

    model->emb_output = (float*)current_ptr;
    current_ptr += b * t * c;

    model->fln_output = (float*)current_ptr;
    current_ptr += b * t * c;

    model->fln_mean = (float*)current_ptr;
    current_ptr += b * t;

    model->fln_std = (float*)current_ptr;
    current_ptr += b * t;

    model->fl_output = (float*)current_ptr;
    current_ptr += b * t * c;

    model->demb_output = (float*)current_ptr;
    current_ptr += b * t * c;

    model->dfln_output = (float*)current_ptr;
    current_ptr += b * t * c;

    model->dfl_output = (float*)current_ptr;
    current_ptr += b * t * c;
}
void unmap_foo(struct foo* model, int b, int t, int c) {
    // Unmap file
    float* map = (float*)model->input; // Assuming input points to the start of the mapped file
    size_t total_size = b * (
        sizeof(int) * t +              // input
        sizeof(float) * t * c +        // emb_output
        sizeof(float) * t * c +        // fln_output
        sizeof(float) * t +            // fln_mean
        sizeof(float) * t +            // fln_std
        sizeof(float) * t * c +        // fl_output
        sizeof(float) * t * c +        // demb_output
        sizeof(float) * t * c          // dfl_output
    );

    // Unmap the memory
    unmap_file(map, total_size);
}

