#!/usr/bin/python3
# -*- coding: utf-8 -*-

from datetime import datetime
import pprint
import copy

import torch
import torch.nn as nn
from model.deepqnet import DeepQNetwork

from hydra import compose, initialize
import hydra
import os
from tensorboardX import SummaryWriter
from collections import deque
from random import random, randint, sample
import numpy as np

class Block_Controller(object):

    # init parameter
    board_backboard = 0
    board_data_width = 0
    board_data_height = 0
    ShapeNone_index = 0
    CurrentShape_class = 0
    NextShape_class = 0

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    # GetNextMove is main function.
    # input
    #    nextMove : nextMove structure which is empty.
    #    GameStatus : block/field/judge/debug information.
    #                 in detail see the internal GameStatus data.
    # output
    #    nextMove : nextMove structure which includes next shape position and the other.

    def __init__(self):
        cfg = self.hydra_read()

        os.makedirs(cfg.common.dir,exist_ok=True)
        self.saved_path = cfg.common.dir + "/" + cfg.common.weight_path
        os.makedirs(self.saved_path ,exist_ok=True)
        self.writer = SummaryWriter(cfg.common.dir+"/"+cfg.common.log_path)

        self.log = cfg.common.dir+"/log.txt"

        self.state_dim = cfg.state.dim

        with open(self.log,"w") as f:
            print("start...", file=f)
        if cfg.model.name=="DQN":
            self.model = DeepQNetwork(self.state_dim )

        self.lr = cfg.train.lr
        if cfg.train.optimizer=="Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()
            state = state.cuda()

        self.mode = cfg.common.mode

        if self.mode!="train":
            self.model = torch.load(cfg.common.load_weight)

        self.replay_memory_size = cfg.train.replay_memory_size
        self.replay_memory = deque(maxlen=self.replay_memory_size)
        self.criterion = nn.MSELoss()

        self.initial_epsilon = 1
        self.final_epsilon = 1e-3
        self.num_decay_epochs = 2000

        self.epoch = 0
        self.num_epochs = 3000

        self.save_interval = 100

        self.height = 22
        self.width = 10
        self.batch_size = 512

        self.score = 0
        self.cleared_lines = 0
        self.gamma = 0.99
        self.iter = 0


        if self.state_dim  ==5:
            self.state = torch.FloatTensor([0,0,0,0,0])
        else:
            self.state = torch.FloatTensor([0,0,0,0])
        self.tetrominoes = 0
        self.penalty = -1
    #[self.state, reward, next_state]
    def update(self):
        if self.mode=="train":
            self.score -= 2
            self.replay_memory[-1][1] = self.penalty
            if len(self.replay_memory) < self.replay_memory_size / 10:
                print("================pass================")
                print("iter: {} ,meory: {}/{} , score: {}, clear line: {}, block: {} ".format(self.iter,
                len(self.replay_memory),self.replay_memory_size / 10,self.score,self.cleared_lines
                ,self.tetrominoes ))
                print("====================================")
            else:
                print("---update---")
                self.epoch += 1
                batch = sample(self.replay_memory, min(len(self.replay_memory),self.batch_size))
                state_batch, reward_batch, next_state_batch = zip(*batch)
                state_batch = torch.stack(tuple(state for state in state_batch))
                reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
                next_state_batch = torch.stack(tuple(state for state in next_state_batch))
                q_values = self.model(state_batch)
                self.model.eval()
                with torch.no_grad():
                    next_prediction_batch = self.model(next_state_batch)

                self.model.train()
                y_batch = torch.cat(
                    tuple(reward if reward<0 else reward + self.gamma * prediction for reward, prediction in
                          zip(reward_batch, next_prediction_batch)))[:, None]

                self.optimizer.zero_grad()
                loss = self.criterion(q_values, y_batch)
                loss.backward()
                self.optimizer.step()
                log = "Epoch: {} / {}, Score: {},  block: {},  Cleared lines: {}".format(
                    self.epoch,
                    self.num_epochs,
                    self.score,
                    #final_tetrominoes,
                    self.tetrominoes,
                    self.cleared_lines
                    )
                print(log)
                with open(self.log,"a") as f:
                    print(log, file=f)
            if self.epoch > self.num_epochs:
                with open(self.log,"a") as f:
                    print("finish..", file=f)
                exit()
        else:
            self.epoch += 1
            log = "Epoch: {} / {}, Score: {},  block: {},  Cleared lines: {}".format(
            self.epoch,
            self.num_epochs,
            self.score,
            #final_tetrominoes,
            self.tetrominoes,
            self.cleared_lines
            )
            pass

    def reset_state(self):
            if self.state_dim  ==5:
                self.state = torch.FloatTensor([0,0,0,0,0])
            else:
                self.state = torch.FloatTensor([0,0,0,0])
            self.score = 0
            self.cleared_lines = 0
            self.tetrominoes = 0
    def hydra_read(self):
        initialize(config_path="../config", job_name="tetris")
        cfg = compose(config_name="default")
        return cfg

    def check_cleared_rows(self,board):
        board_new = np.copy(board)
        lines = 0
        empty_line = np.array([0 for i in range(self.width)])
        for y in range(self.height - 1, -1, -1):
            blockCount  = np.sum(board[y])
            if blockCount == self.width:
                lines += 1
                board_new = np.delete(board_new,y,0)
                board_new = np.vstack([empty_line,board_new ])
        #if lines > 0:
        #    self.backBoard = newBackBoard
        return lines,board_new

    def get_bumpiness_and_height(self,board):
        mask = board != 0
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        heights = self.height - invert_heights
        total_height = np.sum(heights)
        currs = heights[:-1]
        nexts = heights[1:]
        diffs = np.abs(currs - nexts)
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height

    #各列の穴の個数を数える
    def get_holes(self, board):
        num_holes = 0
        for i in range(self.width):
            col = board[:,i]
            row = 0
            while row < self.height and col[row] == 0:
                row += 1
            num_holes += len([x for x in col[row + 1:] if x == 0])
        return num_holes

    def get_state_properties(self, board):
        lines_cleared, board = self.check_cleared_rows(board)
        holes = self.get_holes(board)
        bumpiness, height = self.get_bumpiness_and_height(board)

        return torch.FloatTensor([lines_cleared, holes, bumpiness, height])

    def get_state_properties_v2(self, board):
        lines_cleared, board = self.check_cleared_rows(board)
        holes = self.get_holes(board)
        bumpiness, height = self.get_bumpiness_and_height(board)
        max_row = self.get_max_height(board)
        return torch.FloatTensor([lines_cleared, holes, bumpiness, height,max_row])


    def get_max_height(self, board):
        sum_ = np.sum(board,axis=1)
        row = 0
        while row < self.height and sum_[row] ==0:
            row += 1
        return self.height - row

    def get_next_states(self,GameStatus):
        states = {}
        piece_id =GameStatus["block_info"]["currentShape"]["index"]
        next_piece_id =GameStatus["block_info"]["nextShape"]["index"]
        #curr_piece = [row[:] for row in self.piece]
        if piece_id == 5:  # O piece
            num_rotations = 1
        elif piece_id == 1 or piece_id == 6 or piece_id == 7:
            num_rotations = 2
        else:
            num_rotations = 4
        CurrentShapeDirectionRange = GameStatus["block_info"]["currentShape"]["direction_range"]

        for direction0 in range(num_rotations):
            x0Min, x0Max = self.getSearchXRange(self.CurrentShape_class, direction0)
            for x0 in range(x0Min, x0Max):
                # get board data, as if dropdown block
                board = self.getBoard(self.board_backboard, self.CurrentShape_class, direction0, x0)
                board = self.get_reshape_backboard(board)
                if self.state_dim==5:
                    states[(x0, direction0)] = self.get_state_properties_v2(board)
                else:
                    states[(x0, direction0)] = self.get_state_properties(board)

        return states

            #curr_piece = self.rotate(curr_piece)
    def get_reshape_backboard(self,board):
        board = np.array(board)
        reshape_board = board.reshape(self.height,self.width)
        reshape_board = np.where(reshape_board>0,1,0)
        return reshape_board

    def step(self, action):
        x0, direction0 = action
        board = self.getBoard(self.board_backboard, self.CurrentShape_class, direction0, x0)

        board = self.get_reshape_backboard(board)

        #board[-1] = [1 for i in range(self.width)]
        lines_cleared, board = self.check_cleared_rows(board)
        #print(lines_cleared)
        #input()
        score = 1 + (lines_cleared ** 2) * self.width
        self.score += score
        self.cleared_lines += lines_cleared
        self.tetrominoes += 1
        return score

    def GetNextMove(self, nextMove, GameStatus):

        t1 = datetime.now()

        # print GameStatus
        #print("=================================================>")
        self.ind =GameStatus["block_info"]["currentShape"]["index"]
        self.board_backboard = GameStatus["field_info"]["backboard"]
        # default board definition
        self.board_data_width = GameStatus["field_info"]["width"]
        self.board_data_height = GameStatus["field_info"]["height"]
        self.CurrentShape_class = GameStatus["block_info"]["currentShape"]["class"]
        CurrentShapeDirectionRange = GameStatus["block_info"]["currentShape"]["direction_range"]
        self.CurrentShape_class = GameStatus["block_info"]["currentShape"]["class"]
        # next shape info
        NextShapeDirectionRange = GameStatus["block_info"]["nextShape"]["direction_range"]
        self.NextShape_class = GameStatus["block_info"]["nextShape"]["class"]
        self.ShapeNone_index = GameStatus["debug_info"]["shape_info"]["shapeNone"]["index"]


        reshape_backboard = self.get_reshape_backboard(GameStatus["field_info"]["backboard"])



        #self.crr_backboard = np.where(reshape_backboard>0,1,0)
        next_steps = self.get_next_states(GameStatus)
        #next_steps=[]
        if self.mode=="train":
            epsilon = self.final_epsilon + (max(self.num_decay_epochs - self.epoch, 0) * (
                    self.initial_epsilon - self.final_epsilon) / self.num_decay_epochs)
            u = random()
            random_action = u <= epsilon
            #action = [x0,direction0]
            next_actions, next_states = zip(*next_steps.items())

            next_states = torch.stack(next_states)
            if torch.cuda.is_available():
                next_states = next_states.cuda()
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(next_states)[:, 0]

            self.model.train()
            if random_action:
                index = randint(0, len(next_steps) - 1)
            else:
                index = torch.argmax(predictions).item()
            next_state = next_states[index, :]
            action = next_actions[index]
            reward = self.step(action)
            self.replay_memory.append([self.state, reward, next_state])

            #print("===", datetime.now() - t1)
            nextMove["strategy"]["direction"] = action[1]
            nextMove["strategy"]["x"] = action[0]
            nextMove["strategy"]["y_operation"] = 1
            nextMove["strategy"]["y_moveblocknum"] = 1
            #print(nextMove)
            #print("###### SAMPLE CODE ######")
            self.state = next_state
            self.writer.add_scalar('Train/Score', self.score, self.epoch - 1)
            if self.epoch > 0 and self.epoch % self.save_interval == 0:
                torch.save(self.model, "{}/tetris_{}".format(self.saved_path, self.epoch))
        else:
            self.model.eval()
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)
            predictions = self.model(next_states)[:, 0]
            index = torch.argmax(predictions).item()
            action = next_actions[index]
            nextMove["strategy"]["direction"] = action[1]
            nextMove["strategy"]["x"] = action[0]
            nextMove["strategy"]["y_operation"] = 1
            nextMove["strategy"]["y_moveblocknum"] = 1
        return nextMove

        #exit()
        #reward, done = env.step(action, render=True)

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
        score = score - nIsolatedBlocks * 1.0      # try not to make isolated block
        score = score - absDy * 1.0                # try to put block smoothly
        #score = score - maxDy * 0.3                # maxDy
        #score = score - maxHeight * 5              # maxHeight
        #score = score - stdY * 1.0                 # statistical data
        #score = score - stdDY * 0.01               # statistical data

        # print(score, fullLines, nHoles, nIsolatedBlocks, maxHeight, stdY, stdDY, absDy, BlockMaxY)
        return score


BLOCK_CONTROLLER_SAMPLE = Block_Controller()
