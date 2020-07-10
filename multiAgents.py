# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ghostH = 0
        foodH = 0
        scaredH = 0
        # print(successorGameState.getScore())
        # print(successorGameState.getFood)
        if newFood.asList():
            foodDist = min([util.manhattanDistance(v, newPos) for v in (newFood.asList())])
            foodH = 10 / (foodDist + .1)
            if foodDist <= 0:
                foodH += 250
            # print(foodDist)
        for ghost in newGhostStates:
            ghostDist = util.manhattanDistance(ghost.getPosition(), newPos)
            if ghostDist <= 1:
                foodH /= 10
                ghostH -= 500
            # ghostH += (ghostDist)
        for stime in newScaredTimes:
            scaredH += stime * 50
        if scaredH:
            ghostH += 500
        # print(successorGameState.getScore() + ghostH + foodH + scaredH)
        # if util.manhattanDistance(newPos, currentGameState.getPacmanPosition()) == 0:
        #     foodH -= 1000
        return successorGameState.getScore() + ghostH + foodH + scaredH


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        pacmanActions = gameState.getLegalActions()
        pacmanSuccs = [(gameState.generateSuccessor(0, action), action) for action in pacmanActions]
        scores = [(self.score(succ[0], 1, self.depth), succ[1]) for succ in pacmanSuccs]
        bestAction = max(scores, key=lambda score: score[0])[1]
        # print(pacmanActions[0])
        return bestAction
        # util.raiseNotDefined()

    def score(self, gameState, agent, depth):
        if gameState.isWin() or depth <= 0 or gameState.isLose():
            return self.evaluationFunction(gameState)
        elif agent != 0:
            # ghostSuccs = [gameState.generateSuccessor(0, action) for action in gameState.getLegalActions(0)]
            # ghostScores = [self.score(gameState, i + 1, depth - 1) for i in range(len(gameState.getNumAgents() - 1))]
            # ghostSuccs = [gameState.generateSuccessor(agent, action) for action in gameState.getLegalActions(agent)]
            # ghostScores = [self.score(succ, agent, depth) for succ in ghostSuccs]
            # return min(ghostScores)
            bestScore = min
        else:
            # pacmanScores = [self.score(succ, 1, depth) for succ in pacmanSuccs]
            # return max(pacmanScores)
            bestScore = max

        succs = [gameState.generateSuccessor(agent, action) for action in gameState.getLegalActions(agent)]
        agent = (agent + 1) % gameState.getNumAgents()
        if agent == 0:
            depth = depth - 1
        scores = [self.score(succ, agent, depth) for succ in succs]
        # print(bestScore(scores))
        return bestScore(scores)





class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.score(gameState, 0, self.depth)[1]
        # pacmanActions = gameState.getLegalActions()
        # for action in pacmanActions:
        #     gameState
        # pacmanSuccs = [(gameState.generateSuccessor(0, action), action) for action in ]
        # scores = [(self.score(succ[0], 1, self.depth), succ[1]) for succ in pacmanSuccs]
        # bestAction = max(scores, key=lambda score: score[0])[1]
        # return bestAction
        # util.raiseNotDefined()

    def max_val(self, gameState, agent, depth, bestAction, a=float("-inf"), b=float("inf")):
        v = float("-inf")
        actions = gameState.getLegalActions(agent)
        nextAgent = (agent + 1) % gameState.getNumAgents()
        nextDepth = depth
        if nextAgent == 0:
            nextDepth -= 1
        for action in actions:
            newScore = self.score(gameState.generateSuccessor(agent, action), nextAgent, nextDepth, bestAction, a, b)[0]
            if v < newScore:
                v = newScore
                bestAction = action
            if v > b:
                return v, bestAction
            a = max(a, v)
        return v, bestAction

    def min_val(self, gameState, agent, depth, bestAction, a=float("-inf"), b=float("inf")):
        v = float("inf")
        actions = gameState.getLegalActions(agent)
        nextAgent = (agent + 1) % gameState.getNumAgents()
        nextDepth = depth
        if nextAgent == 0:
            nextDepth -= 1
        for action in actions:
            newScore = self.score(gameState.generateSuccessor(agent, action), nextAgent, nextDepth, bestAction, a, b)[0]
            if v > newScore:
                v = newScore
                bestAction = action
            if v < a:
                return v, bestAction
            b = min(b, v)
            # print(b, "val", v)
        return v, bestAction

    def score(self, gameState, agent, depth, action=None, a=float("-inf"), b=float("inf")):
        # print(a, b)
        if gameState.isWin() or depth <= 0 or gameState.isLose():
            return self.evaluationFunction(gameState), action
        elif agent != 0:
            bestScore = self.min_val(gameState, agent, depth, action, a, b)
        else:
            bestScore = self.max_val(gameState, agent, depth, action, a, b)
        return bestScore

        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
