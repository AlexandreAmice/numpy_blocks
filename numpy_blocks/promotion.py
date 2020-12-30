import numpy as np
from numpy_blocks.numpy_block import block_array

def block_like(scalar_arr, block_arr, **kwargs):
    """
    convert an array of scalars into a block array with blocks the size of block array. Can either
    have diagonal entries like the block array otherwise each block will be
    filled completely with the scalar. If the the blocks of block_arr are not square and the style is 'diag',
    the default behavior will be to make the resulting array conformable such that the result can be right multiplied
    by block_arr
    :param scalar_arr:
    :param block_arr:
    :param kwargs:
    style can be either 'diag' or 'full'. 'diag' results in blocks with the scalar on the diagonal only
    while 'full' fills the entire block with the scalar. 'full' does not guarantee conformability of multiplication
    conformable can be either 'r' or 'l'. 'l' conformable ensures that results @ block_arr returns an array of the
    same shape as block_arr while 'r' conformable ensures that block_arr @ results is valid.
    :return:
    """
    if scalar_arr.shape != block_arr.by_block.shape:
        raise ValueError("Incompatible scalar and block array")
    else:
        style = kwargs.get('style', 'diag')
        conformable = kwargs.get('conformable', 'l')
        blocks = np.ndarray(block_arr.by_block.shape, dtype = np.ndarray)
        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[1]):
                if style == 'full':
                    blocks[i,j] = scalar_arr[i,j] * np.ones(block_arr.by_block.block_shapes[i,j])
                elif style == 'diag':
                    if conformable == 'l':
                        blocks[i,j] = scalar_arr[i,j] * np.eye(block_arr.by_block.block_shapes[i,j][0])
                    elif conformable == 'r':
                        blocks[i, j] = scalar_arr[i, j] * np.eye(block_arr.by_block.block_shapes[i, j][1])
                    else:
                        raise KeyError("conformable must be set to either 'l' or 'r'")
                else:
                    raise KeyError("style must be set to either 'diag' or 'full'")

        return block_array(blocks.tolist())

if __name__ == '__main__':
    col_size = [2, 1, 3]
    row_size = [2, 4, 1]
    A11 = np.ones((row_size[0], col_size[0]))
    A12 = 2 * np.ones((row_size[0], col_size[1]))
    A13 = 5 * np.ones((row_size[0], col_size[2]))

    A21 = 3 * np.ones((row_size[1], col_size[0]))
    A22 = 4 * np.ones((row_size[1], col_size[1]))
    A23 = 6 * np.ones((row_size[1], col_size[2]))

    A31 = 11 * np.ones((row_size[2], col_size[0]))
    A32 = 12 * np.ones((row_size[2], col_size[1]))
    A33 = 13 * np.ones((row_size[2], col_size[2]))

    blocks = block_array([[A11, A12, A13],
                          [A21, A22, A23],
                          [A31, A32, A33]])

    scalar_arr = np.array([[1,2,3],
                           [4,5,6],
                           [7,8,9]])

    promote_full = block_like(scalar_arr, blocks, style='full')

    B11 = np.ones((2,2))
    B12 = 4 * np.ones((2, 2))
    B22 = 3*np.ones((2,2))
    B21 = B12.T/2


    blocks = block_array([[B11, B12], [B21,B22]])
    scalar_arr = np.array([[1, 2],
                           [4, 5]])

    promote_diag_l = block_like(scalar_arr, blocks, style = 'diag', conformable='l')
    promote_diag_r = block_like(scalar_arr, blocks, style = 'diag', conformable='r')

    print()