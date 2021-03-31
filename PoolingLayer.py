import numpy as np
def Pooling(image, type, output_row_size, output_col_size):
    row_step = len(image) // output_row_size
    col_step = len(image) // output_col_size
    pooled = np.zeros(output_row_size, output_col_size)
    for row in output_row_size:
        for col in output_col_size:
            if type == 'max':
                pooled[row, col] = np.max(image[row:row+row_step,col:col+col_step])
            if type == 'mean':
                pooled[row, col] = np.mean(image[row:row + row_step, col:col + col_step])