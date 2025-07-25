#include <metal_stdlib>
using namespace metal;

#define TILE_SIZE 8

kernel void mm(
    device const float *mat1 [[ buffer(0) ]],
    device const float *mat2 [[ buffer(1) ]],
    device float *prodMat [[ buffer(2) ]],
    constant uint *dims [[ buffer(3) ]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y;
    uint j = gid.x;

    uint tRow = tid.y;
    uint tCol = tid.x;

    uint mat1Rows = dims[0];
    uint mat1Cols = dims[1];
    uint mat2Cols = dims[2];

    uint row1Offset = i * mat1Cols;

    threadgroup float mat1Tile[TILE_SIZE][TILE_SIZE];
    threadgroup float mat2Tile[TILE_SIZE][TILE_SIZE];

    uint numTiles = (mat1Cols + TILE_SIZE - 1)/TILE_SIZE;

    float val = 0.0;
    for (uint t = 0; t < numTiles; t++) {

        uint mat1Col = (t * TILE_SIZE) + tCol;
        bool c1 = (i < mat1Rows && mat1Col < mat1Cols);
        mat1Tile[tRow][tCol] = (c1 ? mat1[row1Offset + mat1Col] : 0.0f);
 

        uint mat2Row = (t * TILE_SIZE) + tRow;
        bool c2  (mat2Row < mat1Cols && j < mat2Cols);
        mat2Tile[tCol][tRow] = (c2 ? mat2[mat2Row * mat2Cols + j] : 0.0f);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k+=4) {
            packed_float4 a = *((threadgroup packed_float4*)&mat1Tile[tRow][k]);
            packed_float4 b = *((threadgroup packed_float4*)&mat2Tile[tCol][k]);
            val += dot(a, b);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (i < mat1Rows && j < mat2Cols) {
        prodMat[i * mat2Cols + j] = val;
    }
}

kernel void mmT(
    device const float *mat1 [[ buffer(0) ]],
    device const float *mat2 [[ buffer(1) ]],
    device float *prodMat [[ buffer(2) ]],
    constant uint *dims [[ buffer(3) ]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y;
    uint j = gid.x;

    uint tRow = tid.y;
    uint tCol = tid.x;

    uint mat1Rows = dims[0];
    uint mat1Cols = dims[1];
    uint mat2Cols = dims[2];

    uint row1Offset = i * mat1Cols;
    
    threadgroup float mat1Tile[TILE_SIZE][TILE_SIZE];
    threadgroup float mat2Tile[TILE_SIZE][TILE_SIZE];

    uint numTiles = (mat1Cols + TILE_SIZE - 1)/TILE_SIZE;

    float val = 0.0;

    for (uint t = 0; t < numTiles; t++) {

        uint mat1Col = (t * TILE_SIZE) + tCol;
        bool c1 = (i < mat1Rows && mat1Col < mat1Cols);
        mat1Tile[tRow][tCol] = (c1 ? mat1[row1Offset + mat1Col] : 0.0f);

        uint mat2Row = t * TILE_SIZE + tRow;
        bool c2 = (mat2Row < mat1Cols && j < mat2Cols);
        mat2Tile[tCol][tRow] = (c2 ? mat2[j * mat1Cols + mat2Row] : 0.0f);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k+=4) {
            float4 a = *((threadgroup float4*)&mat1Tile[tRow][k]);
            float4 b = *((threadgroup float4*)&mat2Tile[tCol][k]);
            val += dot(a, b);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (i < mat1Rows && j < mat2Cols) {
        prodMat[i * mat2Cols + j] = val;
    }
}

kernel void mTm(
    device const float *mat1 [[ buffer(0) ]],
    device const float *mat2 [[ buffer(1) ]],
    device float *prodMat [[ buffer(2) ]],
    constant uint *dims [[ buffer(3) ]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y;
    uint j = gid.x;

    uint tRow = tid.y;
    uint tCol = tid.x;

    uint mat1Rows = dims[0];
    uint mat1Cols = dims[1];
    uint mat2Cols = dims[2];

    threadgroup float mat1Tile[TILE_SIZE][TILE_SIZE];
    threadgroup float mat2Tile[TILE_SIZE][TILE_SIZE];

    uint numTiles = (mat1Cols + TILE_SIZE - 1)/TILE_SIZE;

    float val = 0.0;

    for (uint t = 0; t < numTiles; t++) {

        uint mat1Col = t * TILE_SIZE + tCol;
        bool c1 = (i < mat1Rows && mat1Col < mat1Cols);
        mat1Tile[tRow][tCol] = (c1 ? mat1[mat1Col * mat1Rows + i] : 0.0f);


        uint mat2Row = (t * TILE_SIZE) + tRow;
        bool c2  (mat2Row < mat1Cols && j < mat2Cols);
        mat2Tile[tCol][tRow] = (c2 ? mat2[mat2Row * mat2Cols + j] : 0.0f);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k+=4) {
            float4 a = *((threadgroup float4*)&mat1Tile[tRow][k]);
            float4 b = *((threadgroup float4*)&mat2Tile[tCol][k]);
            val += dot(a, b);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (i < mat1Rows && j < mat2Cols) {
        prodMat[i * mat2Cols + j] = val;
    }
}

kernel void mTmT(
    device const float *mat1 [[ buffer(0) ]],
    device const float *mat2 [[ buffer(1) ]],
    device float *prodMat [[ buffer(2) ]],
    constant uint *dims [[ buffer(3) ]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y;
    uint j = gid.x;

    uint tRow = tid.y;
    uint tCol = tid.x;

    uint mat1Rows = dims[0];
    uint mat1Cols = dims[1];
    uint mat2Cols = dims[2];

    threadgroup float mat1Tile[TILE_SIZE][TILE_SIZE];
    threadgroup float mat2Tile[TILE_SIZE][TILE_SIZE];

    uint numTiles = (mat1Cols + TILE_SIZE - 1)/TILE_SIZE;

    float val = 0.0;

    for (uint t = 0; t < numTiles; t++) {

        uint mat1Col = t * TILE_SIZE + tCol;
        bool c1 = (i < mat1Rows && mat1Col < mat1Cols);
        mat1Tile[tRow][tCol] = (c1 ? mat1[mat1Col * mat1Rows + i] : 0.0f);

        uint mat2Row = t * TILE_SIZE + tRow;
        bool c2 = (mat2Row < mat1Cols && j < mat2Cols);
        mat2Tile[tCol][tRow] = (c2 ? mat2[j * mat1Cols + mat2Row] : 0.0f);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k+=4) {
            float4 a = *((threadgroup float4*)&mat1Tile[tRow][k]);
            float4 b = *((threadgroup float4*)&mat2Tile[tCol][k]);
            val += dot(a, b);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (i < mat1Rows && j < mat2Cols) {
        prodMat[i * mat2Cols + j] = val;
    }
}

kernel void colSums(
    device const float *mat [[ buffer(0) ]],
    device float *vec [[ buffer (1) ]],
    constant uint *dims [[ buffer(2) ]],
    uint gid [[thread_position_in_grid]]
) {
    uint col = gid;

    uint matRows = dims[0];
    uint matCols = dims[1];

    if (col >= matCols)
        return;

    float val = 0.0;
    for (uint i = 0; i < matRows; i++) {
        val += mat[i * matCols + col];
    }

    vec[col] = val;
}

kernel void addToRows(
    device float *mat [[ buffer(0) ]],
    device const float *vec [[ buffer (1) ]],
    constant uint *dims [[ buffer(2) ]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y;
    uint j = gid.x;

    uint matRows = dims[0];
    uint matCols = dims[1];

    if (i >= matRows || j >= matCols)
        return;

    mat[i * matCols + j] += vec[j];
}