#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

void cffi_window_edges(
    const long long n, //distance at which points dont get included
    long long* edge_idx,
    const int num_elem, //square window size
    const long long* x,
    const long long* y,
    const long long* b,
    bool self_loop,
    long long* edges1,
    long long* edges2
) {
    int edges_idx = 0;
    for (int elem_idx = 0; elem_idx < num_elem; elem_idx++) {
        int lookahead = elem_idx + 1;
        if(self_loop){
            edges1[edges_idx] = (long long)elem_idx;
            edges2[edges_idx] = (long long)elem_idx;
            edges_idx++;
        }
        while(lookahead < num_elem && b[elem_idx] == b[lookahead]){
            if(abs(x[elem_idx] - x[lookahead]) < n && abs(y[elem_idx] - y[lookahead]) < n){
                edges1[edges_idx] = (long long)elem_idx;
                edges2[edges_idx] = (long long)lookahead;
                edges_idx++;
                edges2[edges_idx] = (long long)elem_idx;
                edges1[edges_idx] = (long long)lookahead;
                edges_idx++;
            }
            lookahead += 1L;
        }
    }
    edge_idx[0] = (long long)edges_idx;
}
