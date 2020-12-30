import numpy as np
import numpy.lib.mixins

class block_array(numpy.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, blocks, **kwargs):
        if type(blocks[0]) is not list:
            blocks = [blocks]

        self.block_list = blocks

        if len(blocks) == 1:
            #handle 1 block type
            if len(blocks[0]) == 1:
                self.blocks = np.ndarray((1,1), dtype = np.ndarray)
                self.blocks[0,0] = blocks[0][0]
            # with one-d array, numpy has trouble distinguishing row or column construction
            else:
                try:
                    self.blocks = np.array(blocks, dtype=np.ndarray, ndmin=2)
                except ValueError:
                    # TODO clean up this cases
                    blocks = [b.T for b in blocks[0]]
                    self.blocks = np.array(blocks, dtype=np.ndarray, ndmin=2)
                    self.blocks = self.blocks.copy()
                    for c in range(self.blocks.shape[1]):
                        self.blocks[0, c] = self.blocks[0, c].T
        else:
            self.blocks = np.array(blocks, dtype=np.ndarray, ndmin=2)

        self.block_index_manager = BlockIndexManager(self.blocks, **kwargs)
        self.block_list = self.block_index_manager.blocks.tolist()
        try:
            self.block_arr = np.block(self.block_list)
        except ValueError as e:
            raise ValueError("Incompatible block types. Check that blocks can be assembled into a proper block array."
                             "Shapes are \n{}".format(self.block_index_manager.block_shapes))



    @classmethod
    def _from_array_and_shapes(cls, array, shape_arr):
        blocks = np.copy(shape_arr)
        cum_row = 0
        for r in range(shape_arr.shape[0]):
            cum_col = 0
            num_rows = shape_arr[r, 0][0]
            for c in range(shape_arr.shape[1]):
                if num_rows != shape_arr[r,c][0]:
                    raise ValueError("Shape array not consistent size")
                num_cols = shape_arr[r,c][1]
                blocks[r,c] = array[cum_row:cum_row+num_rows, cum_col:cum_col + num_cols]
                cum_col += num_cols
            cum_row += num_rows
        return cls(blocks.tolist())

    def __array__(self):
        return np.block(self.block_list)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            if ufunc.__name__ == 'matmul':
                #check conformability
                try:
                    resultant_shapes = self.rmatmul_conformable(inputs[1])
                    arr = inputs[0].__array__() @ inputs[1].__array__()

                    return block_array._from_array_and_shapes(arr, resultant_shapes)
                except NoncomformableException as e:
                    raise Warning("Noncomformable block matrix multiplication, casting to numpy array")
                    return inputs[0].__array__() @ inputs[1].__array__()
            else:
                resultant_shapes = inputs[0].block_index_manager.shapes
                arr = ufunc(inputs[0].__array__(), inputs[1].__array__())
                return block_array._from_array_and_shapes(arr, resultant_shapes)
        else:
            return NotImplemented

    def __array_function__(self, func, types, *args, **kwargs):
        #TODO IMPLEMENT KRON
        pass

    def __getitem__(self, key):
        return self.__array__()[key]

    def __str__(self):
        return "block_"+self.__array__().__str__()

    def __repr__(self):
        return "block_"+self.__array__().__repr__()

    @property
    def by_block(self):
        return self.block_index_manager

    @property
    def shape(self):
        return self.block_arr.shape

    def rmatmul_conformable(self, other):
        if self.blocks.shape[1] != other.blocks.shape[0]:
            raise BlockMatrixNoncomformableExcpetion(self.blocks, other.blocks)
        if self.block_index_manager.block_shapes.shape[0] != other.block_index_manager.shapes.shape[1]:
            raise BlockMatrixNoncomformableExcpetion
        #TODO parallelize
        non_conformable_block_pairs = []
        non_conformable_block_inds = []
        resultant_shapes = np.ndarray((self.blocks.shape[0], other.blocks.shape[1]), dtype = object)
        for r1 in range(self.block_index_manager.block_shapes.shape[0]):
            for c2 in range(other.block_index_manager.shapes.shape[1]):
                for j in range(self.block_index_manager.block_shapes.shape[1]):
                    s1 = self.block_index_manager.block_shapes[r1, j]
                    s2 = other.block_index_manager.shapes[j, c2]
                    if s1[1] != s2[0]:
                        non_conformable_block_inds.append(((r1,j),(j,c2)))
                        non_conformable_block_pairs.append((self.blocks[r1, j], other.blocks[j,c2]))
                    else:
                        resultant_shapes[r1,c2] = (s1[0], s2[1])
        if non_conformable_block_inds:
            raise BlockPartitionNoncomformableException(non_conformable_block_pairs, non_conformable_block_inds)
        else:
            return resultant_shapes

    def promote_scalar_blocks(self, block_list, index_manager):
        """
        promotes scalar blocks to be identity with that value in the diagonal.
        :param block_list:
        :return:
        """
        for i in range(len(block_list)):
            for j in range(len(block_list[i])):
                try:
                    assert block_list[i][j].shape == index_manager.shapes[i,j]
                except AttributeError:
                    if index_manager.shapes[i,j][0] == index_manager.shapes[i,j][1]:
                        block_list[i][j] = block_list[i][j] * np.eye(index_manager.shapes[i,j][0])
                    else:
                        block_list[i][j] = block_list[i][j] * np.ones(index_manager.shapes[i, j])
                return block_list


class BlockIndexManager(object):
    def __init__(self, blocks, **kwargs):
        self.blocks = blocks.view()
        self.block_shapes = np.ndarray((len(blocks), len(blocks[0])), dtype = object)
        self.global_inds = np.ndarray((len(blocks), len(blocks[0])), dtype = object)

        row_cum_sum = 0

        revisit_index = []
        row_size = [-1 for _ in range(len(blocks))]
        col_size = [-1 for _ in range(len(blocks[0]))]

        for i in range(len(blocks)):
            row_size_specified = False
            for j in range(len(blocks[i])):
                try:
                    row_size[i] = blocks[i,j].shape[0]
                    col_size[j] = blocks[i,j].shape[1]
                    row_size_specified = True
                    continue
                except AttributeError:
                    pass
            if not row_size_specified:
                row_size[i] = 1

        for i in range(len(blocks)):
            col_cum_sum = 0
            row_stride = 0
            for j in range(len(blocks[i])):
                try:
                    self.block_shapes[i, j] = blocks[i, j].shape
                except AttributeError as e:
                    if kwargs.get('promote_scalar_blocks', True):
                        self.block_shapes[i, j] = (row_size[i], col_size[j])
                        if self.block_shapes[i, j][0] == self.block_shapes[i, j][1]:
                            blocks[i,j] = blocks[i,j] * np.eye(self.block_shapes[i, j][0])
                        else:
                            blocks[i,j] = blocks[i,j] * np.ones(self.block_shapes[i, j])
                    else:
                        raise e
                row_stride = blocks[i][0].shape[0]
                if j > 0 and self.block_shapes[i, j - 1][0] != self.block_shapes[i, j][0]:
                    raise ValueError("Incompatible blocks")
                col_stride = blocks[i][j].shape[1]
                self.global_inds[i,j] = np.s_[row_cum_sum:row_cum_sum+row_stride, col_cum_sum : col_cum_sum+col_stride]
                col_cum_sum += col_stride
            row_cum_sum += row_stride

    @classmethod
    def global_inds_from_shape(cls, shapes):
        global_inds = np.ndarray(shapes.shape, dtype=object)
        row_cum_sum = 0
        for i in range(shapes.shape[0]):
            col_cum_sum = 0
            row_stride = shapes.shape[0]
            for j in range(shapes.shape[1]):
                col_stride = shapes[i][j].shape[1]
                global_inds[i, j] = np.s_[row_cum_sum:row_cum_sum + row_stride,
                                         col_cum_sum: col_cum_sum + col_stride]
                col_cum_sum += col_stride
            row_cum_sum += row_stride
        return global_inds

    def __getitem__(self, item):
        b = self.blocks[item]
        return block_array(b.tolist()) if type(b[0]) is list else block_array([b])

    @property
    def shape(self):
        return self.block_shapes.shape

class NoncomformableException(Exception):
    pass

class BlockMatrixNoncomformableExcpetion(NoncomformableException):
    def __init__(self, mat1, mat2):
        self.mat1 = mat1
        self.mat2 = mat2
        self.message = "Not comformable, size {} x {}".format(mat1.shape, mat2.shape)
        super().__init__(self.message)

class BlockPartitionNoncomformableException(NoncomformableException):
    def __init__(self, block_pairs, block_ind_pairs):
        self.block_pairs = block_pairs
        self.block_ind_pairs = block_ind_pairs
        self.message = ""
        for i in range(min(len(self.block_pairs),10)):
            block_1 = self.block_pairs[i][0]
            block_2 = self.block_pairs[i][1]
            block_ind1 = self.block_ind_pairs[i][0]
            block_ind2 = self.block_ind_pairs[i][1]
            self.message += "Non conformable block: size {} x {} at index {} and {}\n".format(block_1.shape, block_2.shape, block_ind1, block_ind2)
        super().__init__(self.message)


if __name__ == '__main__':

    C11 = np.eye(2,)
    C22 = 5*np.ones((5,5))
    blocks = block_array([[C11, 0],[0, C22]])
    print(blocks)

    # B11 = np.ones((2,2))
    # B12 = 4 * np.ones((2, 1))
    # B22 = 3*np.ones((1,1))
    # B21 = B12.T/2
    #
    #
    # blocks = block_array([[B11, B12], [B21,B22]])
    # print()
    # blocks @blocks
    #
    #
    col_size = [2,1,3]
    row_size = [2,4,1]
    A11 = np.ones((row_size[0], col_size[0]))
    A12 = 2*np.ones((row_size[0], col_size[1]))
    A13 = 5 * np.ones((row_size[0], col_size[2]))

    A21 = 3*np.ones((row_size[1], col_size[0]))
    A22 = 4*np.ones((row_size[1], col_size[1]))
    A23 = 6*np.ones((row_size[1], col_size[2]))

    A31 = 11*np.ones((row_size[2], col_size[0]))
    A32 = 12*np.ones((row_size[2], col_size[1]))
    A33 = 13*np.ones((row_size[2], col_size[2]))



    # blocks = block_array([[A11, A12, A13],
    #                       [A21, A22, A23],
    #                       [A31, A32, A33]])
    blocks = block_array([[A11, A12, A13]])

    np_arr = np.array([[A11, A12, A13],
                          [A21, A22, A23],
                          [A31, A32, A33]], dtype = object)


    # b2 = blocks@blocks

    print(blocks.by_block[0, 0])
    # print(blocks.by_block[:, 0])
    print(blocks.by_block[0, :])
    print(blocks.by_block[0:1, :])
    print(blocks.by_block[:, 0])
    print(blocks.by_block[:, 0:1])
    print(blocks.by_block[1:2, :])