#!/usr/bin/python3
# -*- coding: utf-8 -*-

from datetime import datetime
import pprint
import copy
import numpy as np
class Block_Controller(object):

    # init parameter
    board_backboard = 0
    board_data_width = 0
    board_data_height = 0
    ShapeNone_index = 0
    CurrentShape_class = 0
    NextShape_class = 0
    def __init__(self):
        #state_dim =  #direction_range
        self.max_point = 1300
        self.get_board_width = 10
        self.get_board_height = 5
        self.direction_dim = 4
        self.condition1_dim = 2 #current ,next
        self.condition2_dim = self.condition1_dim  + 1  #current ,next,direction

        self.filed_dim = self.get_board_width * self.get_board_height

        self.observation1_space_dim = self.condition1_dim + self.filed_dim
        self.observation2_space_dim = self.condition2_dim + self.filed_dim


        self.action_space_1_dim = self.direction_dim #decide a direction
        self.action_space_2_dim = self.get_board_width #decide a x


        #initialize_array
        self.condition_vec1 =np.zeros(self.condition1_dim)
        self.condition_vec2 =np.zeros(self.condition2_dim)

        self.observation1_space = np.zeros([self.observation1_space_dim,1]) #52,1
        self.observation2_space = np.zeros([self.observation2_space_dim,1]) #53,1

        self.param_action1 = np.random.rand(self.observation1_space_dim, self.action_space_1_dim)  #52x4
        self.param_action2 = np.random.rand(self.observation2_space_dim, self.action_space_2_dim)  #53x10

        self.action1_list = []
        self.action2_list = []
        self.reward_list = []
        self.score_list = []

        self.score = 0
        self.episode_iter = 0
        self.episode_num = 0
        self.episode_reward = 0.0
        self.line_score = 0.0
    # GetNextMove is main function.
    # input
    #    nextMove : nextMove structure which is empty.
    #    GameStatus : block/field/judge/debug information.
    #                 in detail see the internal GameStatus data.
    # output
    #    nextMove : nextMove structure which includes next shape position and the other.
    def GetNextMove(self, nextMove, GameStatus):

        t1 = datetime.now()
        board_width = GameStatus["field_info"]["width"]
        board_height = GameStatus["field_info"]["height"]
        game_over_count = GameStatus["judge_info"]["gameover_count"]
        current_shape_index =GameStatus["block_info"]["currentShape"]["index"]
        next_shape_index =GameStatus["block_info"]["nextShape"]["index"]
        reshape_backboard = self.get_reshape_backboard(GameStatus["field_info"]["backboard"],board_height,board_width)
        reshape_backboard = np.where(reshape_backboard>0,1,0)
        reshape_backboard_nlines = self.get_backboard_n_lines(reshape_backboard,self.get_board_height)
        backboard_nlines = reshape_backboard_nlines.flatten()

        #update parameter
        if self.episode_num < game_over_count:
            print(">>>>Change episode")
            self.episode_num = game_over_count
            self.episode_score = 0.0
            self.episode_iter = 0
        prev_point = GameStatus["debug_info"]["linescore"] - self.line_score
        prev_reward =prev_point/self.max_point

        if self.episode_iter >0:
            self.episode_reward += prev_reward
            self.line_score = GameStatus["debug_info"]["linescore"]


        # print GameStatus
        print("=================================================>")
        #pprint.pprint(GameStatus, width = 61, compact = True)
        print("epsiode num: %d"%(self.episode_num ))
        print("epsiode iter: %d"%(self.episode_iter))
        print("reward: %f"%(prev_reward))
        print("line score %f"%(GameStatus["debug_info"]["linescore"]))
        print("backboard_nlines:")
        print(reshape_backboard_nlines)
        #print(GameStatus["field_info"]["backboard"])
        print("field_shape:")
        #print(board_height,board_width)
        print("current_shape_index %d"%(current_shape_index))
        print("next_shape_index %d"%(next_shape_index))
        #print(reshape_backboard_nlines)

        self.condition_vec2[0] = current_shape_index
        self.condition_vec2[1] = next_shape_index

        # get data from GameStatus
        # current shape info
        CurrentShapeDirectionRange = GameStatus["block_info"]["currentShape"]["direction_range"]
        self.CurrentShape_class = GameStatus["block_info"]["currentShape"]["class"]
        # next shape info
        NextShapeDirectionRange = GameStatus["block_info"]["nextShape"]["direction_range"]
        self.NextShape_class = GameStatus["block_info"]["nextShape"]["class"]
        # current board info
        self.board_backboard = GameStatus["field_info"]["backboard"]
        # default board definition
        self.board_data_width = GameStatus["field_info"]["width"]
        self.board_data_height = GameStatus["field_info"]["height"]
        self.ShapeNone_index = GameStatus["debug_info"]["shape_info"]["shapeNone"]["index"]

        # search best nextMove -->
        strategy = None
        LatestEvalValue = -100000
        # search with current block Shape

        for direction0 in CurrentShapeDirectionRange:
            self.condition_vec2[2] = direction0
            backboard_nlines_with_condition =np.append(self.condition_vec2,backboard_nlines) #1x53
            self.observation2_space = backboard_nlines_with_condition #reshape(-1,1)
            probs = self.get_policy(self.observation2_space,self.param_action2)
            action = np.random.choice(self.action_space_2_dim ,1,p=probs)[0]
            #print(probs,action)

            # search with x range
            x0Min, x0Max = self.getSearchXRange(self.CurrentShape_class, direction0)


            for x0 in range(x0Min, x0Max):
                # get board data, as if dropdown block
                board = self.getBoard(self.board_backboard, self.CurrentShape_class, direction0, x0)
                reshape_backboard_next = self.get_reshape_backboard(board,board_height,board_width)
                reshape_backboard_next = np.where(reshape_backboard_next>0,1,0)
                #print(reshape_backboard_next)
                eval_board_h= self.eval_continuous_block_horizontal(reshape_backboard_next)
                #print(eval_board_h)
                print(np.mean(eval_board_h))
                #score_next = self.calcEvaluationValue(reshape_backboard_next)
                # evaluate board
                EvalValue = self.calcEvaluationValueSample(board)
                print(EvalValue)

                # update best move
                if EvalValue > LatestEvalValue:
                    strategy = (direction0, x0, 1, 1)
                    LatestEvalValue = EvalValue

        # search best nextMove <--
        print("===", datetime.now() - t1)
        nextMove["strategy"]["direction"] = strategy[0]
        nextMove["strategy"]["x"] = strategy[1]
        nextMove["strategy"]["y_operation"] = strategy[2]
        nextMove["strategy"]["y_moveblocknum"] = strategy[3]
        print(nextMove)
        print("###### SAMPLE CODE ######")
        self.episode_iter += 1
        return nextMove

    def eval_continuous_block_horizontal(self,board):
        #board = np.random.randint(0,2,(5,12))
        h,w = board.shape
        right_shift = np.roll(board,1)
        right_shift[:,0]=board[:,0]
        left_shift = np.roll(board,-1)
        left_shift[:,-1]=board[:,-1]
        eval_board =  (board + right_shift + left_shift)/3.0
        eval_board[board==0] =0
        return eval_board


    def calcEvaluationValue(self,board):
        score = 0
        board_sum = np.sum(board,axis=1)


    def softmax(self,x):
        x = np.exp(x - np.max(x))
        return x / x.sum()

    def get_policy(self,state, theta):
        z = np.dot(theta.T,state)
        return self.softmax(z)

    def get_reshape_backboard(self,board,height,width):
        board = np.array(board)
        reshape_board = board.reshape(height,width)
        return reshape_board

    def get_backboard_n_lines(self,board,n):
        board_one_line = np.sum(board,axis=1)
        h,w = board.shape
        index = np.where(board_one_line>0)[0]
        if len(index)==0:
            board_nline = board[h-n:h,:]
        else:

            top_index=index[0]
            top_index = h-n if top_index+n > h else top_index
            board_nline = board[top_index:top_index+n,:]
        return board_nline

    def getSearchXRange(self, Shape_class, direction):
        #
        # get x range from shape direction.
        #
        minX, maxX, _, _ = Shape_class.getBoundingOffsets(direction) # get shape x offsets[minX,maxX] as relative value.
        xMin = -1 * minX
        xMax = self.board_data_width - maxX
        return xMin, xMax

    def getShapeCoordArray(self, Shape_class, direction, x, y):
        #
        # get coordinate array by given shape.
        #
        coordArray = Shape_class.getCoords(direction, x, y) # get array from shape direction, x, y.
        return coordArray

    def getBoard(self, board_backboard, Shape_class, direction, x):
        #
        # get new board.
        #
        # copy backboard data to make new board.
        # if not, original backboard data will be updated later.
        board = copy.deepcopy(board_backboard)
        _board = self.dropDown(board, Shape_class, direction, x)
        return _board


    def dropDown(self, board, Shape_class, direction, x):
        #
        # internal function of getBoard.
        # -- drop down the shape on the board.
        #
        dy = self.board_data_height - 1
        coordArray = self.getShapeCoordArray(Shape_class, direction, x, 0)
        # update dy
        for _x, _y in coordArray:
            _yy = 0
            while _yy + _y < self.board_data_height and (_yy + _y < 0 or board[(_y + _yy) * self.board_data_width + _x] == self.ShapeNone_index):
                _yy += 1
            _yy -= 1
            if _yy < dy:
                dy = _yy
        # get new board
        _board = self.dropDownWithDy(board, Shape_class, direction, x, dy)
        return _board

    def dropDownWithDy(self, board, Shape_class, direction, x, dy):
        #
        # internal function of dropDown.
        #
        _board = board
        coordArray = self.getShapeCoordArray(Shape_class, direction, x, 0)
        for _x, _y in coordArray:
            _board[(_y + dy) * self.board_data_width + _x] = Shape_class.shape
        return _board

    def calcEvaluationValueSample(self, board):
        #
        # sample function of evaluate board.
        #
        width = self.board_data_width
        height = self.board_data_height

        # evaluation paramters
        ## lines to be removed
        fullLines = 0
        ## number of holes or blocks in the line.
        nHoles, nIsolatedBlocks = 0, 0
        ## absolute differencial value of MaxY
        absDy = 0
        ## how blocks are accumlated
        BlockMaxY = [0] * width
        holeCandidates = [0] * width
        holeConfirm = [0] * width

        ### check board
        # each y line
        for y in range(height - 1, 0, -1):
            hasHole = False
            hasBlock = False
            # each x line
            for x in range(width):
                ## check if hole or block..
                if board[y * self.board_data_width + x] == self.ShapeNone_index:
                    # hole
                    hasHole = True
                    holeCandidates[x] += 1  # just candidates in each column..
                else:
                    # block
                    hasBlock = True
                    BlockMaxY[x] = height - y                # update blockMaxY
                    if holeCandidates[x] > 0:
                        holeConfirm[x] += holeCandidates[x]  # update number of holes in target column..
                        holeCandidates[x] = 0                # reset
                    if holeConfirm[x] > 0:
                        nIsolatedBlocks += 1                 # update number of isolated blocks

            if hasBlock == True and hasHole == False:
                # filled with block
                fullLines += 1
            elif hasBlock == True and hasHole == True:
                # do nothing
                pass
            elif hasBlock == False:
                # no block line (and ofcourse no hole)
                pass

        # nHoles
        for x in holeConfirm:
            nHoles += abs(x)

        ### absolute differencial value of MaxY
        BlockMaxDy = []
        for i in range(len(BlockMaxY) - 1):
            val = BlockMaxY[i] - BlockMaxY[i+1]
            BlockMaxDy += [val]
        for x in BlockMaxDy:
            absDy += abs(x)

        #### maxDy
        #maxDy = max(BlockMaxY) - min(BlockMaxY)
        #### maxHeight
        #maxHeight = max(BlockMaxY) - fullLines

        ## statistical data
        #### stdY
        #if len(BlockMaxY) <= 0:
        #    stdY = 0
        #else:
        #    stdY = math.sqrt(sum([y ** 2 for y in BlockMaxY]) / len(BlockMaxY) - (sum(BlockMaxY) / len(BlockMaxY)) ** 2)
        #### stdDY
        #if len(BlockMaxDy) <= 0:
        #    stdDY = 0
        #else:
        #    stdDY = math.sqrt(sum([y ** 2 for y in BlockMaxDy]) / len(BlockMaxDy) - (sum(BlockMaxDy) / len(BlockMaxDy)) ** 2)


        # calc Evaluation Value
        score = 0
        score = score + fullLines * 10.0           # try to delete line
        score = score - nHoles * 1.0               # try not to make hole
        #score = score - nIsolatedBlocks * 1.0      # try not to make isolated block
        score = score - absDy * 1.0                # try to put block smoothly
        #score = score - maxDy * 0.3                # maxDy
        #score = score - maxHeight * 5              # maxHeight
        #score = score - stdY * 1.0                 # statistical data
        #score = score - stdDY * 0.01               # statistical data

        # print(score, fullLines, nHoles, nIsolatedBlocks, maxHeight, stdY, stdDY, absDy, BlockMaxY)
        return score


BLOCK_CONTROLLER_SAMPLE = Block_Controller()
