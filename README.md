#### gpt2 written in plain C

The main goal of this small project is to educate myself on how things are built from scratch, and I hope to convince at least a single person that they could build anything from scratch. Andrej Karpathy's llm.c and micrograd were the projects that motivated me to build this.

#### things I learnt building this:

1. Multi-dimensional arrays and tensors are just simple 1-dimensional arrays but with strides enabling us to access rows and columns in the desired way.
2. Learned a lot about the C language, including memory management, parallel processing, and memory access patterns. This is just the second thing I built in C, the first one being a basic password manager.
3. Derived backpropagation of layers like LayerNorm and Attention mechanisms. Improved my mathematical ability a lot.
4. Learned about how we could map files and use them as a sort of virtual memory (it was hard storing the activations and parameters in the RAM. They are humongous, something like ~20GB).

It was fun building something like this.

#### compiler flags

Some compiler flags to optimize the performance: -O3 -march=native -funroll-loops -fopenmp

- `-O3`: Aggressive optimizations
- `-march=native`: CPU-specific optimizations
- `-funroll-loops`: Loop unrolling for potential speed improvements
- `-fopenmp`: OpenMP support for parallel processing

#### blogs that helped me a lot:
- [Matrix Multiplication on CPU](https://marek.ai/matrix-multiplication-on-cpu.html)
- [Fast MMM on CPU](https://siboehm.com/articles/22/Fast-MMM-on-CPU)

This implementation isn't the most optimal approach; there are lots of things to improve.

#### references:
- [gpt in python](https://github.com/shaRk-033/GPT_MINI)

#### TODO:
- Improve the Matrix Multiplication.
- Improve the Attention Mechanism and its backprop, as it consumes a lot of training time.
