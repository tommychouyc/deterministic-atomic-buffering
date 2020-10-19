typedef struct config config;

struct config
{
    int i_n; 
    int i_c;
    int i_h;
    int i_w;
  
    int o_n; 
    int o_c;
    int o_h;
    int o_w;

    int f_k;
    int f_c;
    int f_h;
    int f_w;

    int pad_h;
    int stride_h;
    int u;

    int pad_w;
    int stride_w;
    int v;
};
