# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
from copy import deepcopy

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, **kwargs):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [OffensiveAgent(firstIndex, **kwargs), DefensiveAgentWithOffenseTactic(secondIndex)]

##########
# Agents #
##########


class ReinforcementAgent(CaptureAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        if kwargs.get('numTraining') is not None:
            self.numTraining = int(kwargs['numTraining'])
        else:
            self.numTraining = 0

        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.epsilon = 0.0
        self.alpha = 0.0
        self.discount = 0.9

        self.weights = {
            'in-danger': 0,
            'eats-food': 275.7072069251819,
            'eats-capsule': 300.7072069251819,
            'closest-food': -3.0241968966220902,
            'closest-capsule': -5,
            'closest-border': -3.02419689662209021,
            'returns-food': 400,
            'in-dead-end': -982.1249192391,
            'stop': -100,
            'dist-to-ghost': 5.1,
            'dist-to-best-entry': -3.025,
            'turn-defense': -2,
            'turn-offense': 1,
            'eaten-by-ghost': -10000,
        }
        self.previousActions = []

    def getLegalActions(self, gameState):
        return gameState.getLegalActions(self.index)

    def observeTransition(self, gameState, action, nextGameState, reward):
        self.episodeRewards += reward
        self.update(gameState, action, nextGameState, reward)

    def startEpisode(self):
        self.lastGameState = None
        self.lastAction = None
        self.episodeRewards = 0.0

    def stopEpisode(self):
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0    # no exploration
            self.alpha = 0.0      # no learning

    def isInTraining(self):
        return self.episodesSoFar < self.numTraining

    def isInTesting(self):
        return not self.isInTraining()

    def calculateReward(self, gameState):
        return gameState.getScore()

    ################################
    #          Q-Learning          #
    ################################
    def update(self, gameState, action, nextGameState, reward):
        features = self.getFeatures(gameState, action)
        for feature, value in features.items():
            difference = (reward + self.discount * self.getValue(nextGameState)) - self.getQValue(gameState, action)
            self.weights[feature] = self.weights[feature] + self.alpha * difference * value

    def getFeatures(self, gameState, action):
        features = util.Counter()
        return features

    def getWeights(self):
        return self.weights

    def getQValue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights()
        return features * weights

    def computeValueFromQValues(self, gameState):
        actions = self.getLegalActions(gameState)
        if len(actions) > 0:
            return max([self.getQValue(gameState, action) for action in actions])
        return 0.0

    def computeActionFromQValues(self, gameState):
        actions = self.getLegalActions(gameState)
        maxValue = self.computeValueFromQValues(gameState)
        bestActions = [action for action in actions if self.getQValue(gameState, action) == maxValue]
        if len(bestActions) > 0:
            return random.choice(bestActions)
        elif len(actions) > 0:
            return random.choice(actions)
        else:
            return None

    def getPolicy(self, gameState):
        return self.computeActionFromQValues(gameState)

    def getValue(self, gameState):
        return self.computeValueFromQValues(gameState)

    def doAction(self, state, action):
        """
            Called by inherited class when
            an action is taken in a state
        """
        self.lastGameState = state
        self.lastAction = action
        self.previousActions.append(action)

    ####################
    #  Agent Specific  #
    ####################
    def checkLoopAction(self, gameState):
        if len(self.previousActions) < 4:
            return None
        if self.previousActions[-1] == self.previousActions[-3] and self.previousActions[-2] == self.previousActions[-4]\
                and self.previousActions[-1] != self.previousActions[-2] and self.previousActions[-3] != self.previousActions[-4]:
            print('loop action: {}'.format(self.previousActions[-2]))
            loopAction = self.previousActions[-2]
            self.previousActions = []
            return loopAction
        else:
            return None

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getMyState(self, gameState):
        return gameState.getAgentState(self.index)

    def getMyPosition(self, gameState):
        return self.getMyState(gameState).getPosition()

    def getOpponentsState(self, gameState):
        return [gameState.getAgentState(i) for i in self.getOpponents(gameState)]

    def observationFunction(self, gameState):
        if not self.lastGameState is None:
            reward = self.calculateReward(gameState)
            self.observeTransition(self.lastGameState, self.lastAction, gameState, reward)

        return gameState.makeObservation(self.index)

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.startEpisode()
        self.startPos = gameState.getAgentPosition(self.index)
        if self.episodesSoFar == 0:
            print('Beginning %d episodes of Training' % (self.numTraining))

    def final(self, gameState):
        """
          Called by Pacman game at the terminal state
        """
        reward = self.calculateReward(gameState)
        self.observeTransition(self.lastGameState, self.lastAction, gameState, reward)
        self.stopEpisode()

        if self.episodesSoFar == self.numTraining:
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg,'-' * len(msg)))
            print(self.getWeights())

    def chooseAction(self, gameState):
        print('---------------------------------')
        # Pick Action
        legalActions = self.getLegalActions(gameState)

        loopAction = self.checkLoopAction(gameState)
        if loopAction is not None and loopAction in legalActions:
            legalActions.remove(loopAction)
            return random.choice(legalActions)

        if len(legalActions) == 0:
            return None
        if util.flipCoin(self.epsilon):
            selectedAction = random.choice(legalActions)
        else:
            selectedAction = self.getPolicy(gameState)

        self.doAction(gameState, selectedAction)
        for a in legalActions:
            print(a, self.getFeatures(gameState, a))
        print('Action taken: {}'.format(selectedAction))
        print('---------------------------------')
        return selectedAction


class OffensiveAgent(ReinforcementAgent):
    def registerInitialState(self, gameState):
        ReinforcementAgent.registerInitialState(self, gameState)
        layout = deepcopy(gameState.data.layout)
        mapWidth = layout.width
        mapHeight = layout.height
        if self.red:
            opponentX = int(mapWidth/2)
        else:
            opponentX = int(mapWidth/2) - 1

        self.entries = []
        if self.red:
            borderX = int(mapWidth / 2) - 1
        else:
            borderX = int(mapWidth / 2)
        for borderY in range(0, mapHeight):
            if not layout.isWall((borderX, borderY)):
                self.entries.append((borderX, borderY))

        self.deadEnds = []
        startX = opponentX if self.red else 0
        endX = mapWidth if self.red else opponentX
        while True:
            prevLen = len(self.deadEnds)
            newWalls = deepcopy(layout.walls)
            for x in range (startX, endX):
                for y in range(0, mapHeight):
                    if not layout.walls[x][y] and self.isSurroundByWalls((x, y), layout.walls):
                        self.deadEnds.append((x, y))
                        newWalls[x][y] = True
            layout.walls = newWalls
            if prevLen == len(self.deadEnds):
                break

        self.closestGhost = None
        self.totalFoodNum = len(self.getFood(gameState).asList())
        self.startPos = gameState.getAgentState(self.index).getPosition()
        self.bestEntry = None

    def isSurroundByWalls(self, pos, walls):
        wallsAround = 0
        x, y = pos
        try:
            if walls[x+1][y]:
                wallsAround += 1
        except:
            pass
        try:
            if walls[x-1][y]:
                wallsAround += 1
        except:
            pass
        try:
            if walls[x][y+1]:
                wallsAround += 1
        except:
            pass
        try:
            if walls[x][y-1]:
                wallsAround += 1
        except:
            pass
        return wallsAround >= 3

    def isDeadEnd(self, pos):
        return pos in self.deadEnds

    def getFeatures(self, gameState, action):
        features = util.Counter()

        newGameState = self.getSuccessor(gameState, action)
        myCurState = self.getMyState(gameState)
        myNewState = self.getMyState(newGameState)
        myNewPos = myNewState.getPosition()
        walls = gameState.getWalls()
        x, y = myNewPos
        x = int(x)
        y = int(y)
        foods = self.getFood(gameState)
        capsules = self.getCapsules(gameState)
        opponents = self.getOpponentsState(gameState)

        movesLeft = int (gameState.data.timeleft) / 4

        if action == Directions.STOP: features['stop'] = 1

        # Get the closest ghost
        ghosts = [g for g in opponents if g.getPosition() is not None and not g.isPacman and g.scaredTimer < 1]
        if len(ghosts) > 0:
            minDist = 999999
            for g in ghosts:
                dist = self.getMazeDistance(myNewPos, g.getPosition())
                if dist <= 5 and dist < minDist:
                    minDist = dist
                    self.closestGhost = g

        # If on our side
        if not myNewState.isPacman:
            maxScore = -9999999
            for e in self.entries:
                distToGhost = self.getMazeDistance(e, self.closestGhost.getPosition()) if self.closestGhost else 0
                if len(foods.asList()) > 0:
                    distToFood = min([self.getMazeDistance(e, f) for f in foods.asList()])
                    score = distToGhost - distToFood
                else:
                    score = distToGhost

                if score > maxScore:
                    maxScore = score
                    self.bestEntry = e

            features['dist-to-best-entry'] = float(self.getMazeDistance(myNewPos, self.bestEntry)) / (walls.width * walls.height)

            inDanger = False
            if self.closestGhost and self.getMazeDistance(self.closestGhost.getPosition(), myCurState.getPosition()) <= 2:
                inDanger = True

            if myCurState.isPacman and myNewPos == self.startPos:
                features['eaten-by-ghost'] = 1
            if myCurState.isPacman and myNewPos != self.startPos and myCurState.numCarrying == 0 and not inDanger:
                features['turn-defense'] = 1
            if myCurState.isPacman and myNewPos != self.startPos and myCurState.numCarrying > 0:
                features['returns-food'] = 1.0
        # If on the opponent's side
        else:
            if not myCurState.isPacman and myNewState.isPacman and self.getMazeDistance(self.bestEntry, myCurState.getPosition()) <= 1:
                features['turn-offense'] = 1

            # self.closestGhost will remember the last location of the ghost, even if it has gone far away
            # In here, we only want to take into account the real ghost around us
            closestGhost = None
            ghosts = [g for g in opponents if g.getPosition() is not None and not g.isPacman]
            if len(ghosts) > 0:
                minDist = 999999
                for g in ghosts:
                    dist = self.getMazeDistance(myNewPos, g.getPosition())
                    if dist <= 5 and dist < minDist:
                        minDist = dist
                        closestGhost = g

            if closestGhost and self.getMazeDistance(myNewPos, closestGhost.getPosition()) <= 5 and closestGhost.scaredTimer <= 5:
                distToGhost = self.getMazeDistance(myNewPos, closestGhost.getPosition())
                features['dist-to-ghost'] = float(distToGhost) / (walls.width * walls.height)
                features['in-danger'] = 1
                if distToGhost <= 1:
                    features['eaten-by-ghost'] = 1
                if self.isDeadEnd((x, y)):
                    features['in-dead-end'] = 1
                    # Reverse this so it can move towards the ghost and get out instead of hiding in the dead end
                    features['dist-to-ghost'] = -features['dist-to-ghost']

                distToBorder = min([self.getMazeDistance(myNewPos, e) for e in self.entries])
                if len(capsules) > 0 and movesLeft - 10 >= distToBorder:
                    if myNewPos in capsules:
                        features['eats-capsule'] = 1.0
                    else:
                        distToCapsule = min([self.getMazeDistance(myNewPos, c) for c in capsules])
                        bestCapsule = [c for c in capsules if self.getMazeDistance(myNewPos, c) == distToCapsule][0]
                        features['closest-capsule'] = float(distToCapsule) / (walls.width * walls.height)
                        if self.lastGameState:
                            myLastPos = self.getMyState(self.lastGameState).getPosition()
                            lastDistToCapsule = self.getMazeDistance(myLastPos, bestCapsule)
                            ghostDistToCapsule = self.getMazeDistance(closestGhost.getPosition(), bestCapsule)

                            # If we are trying to get to the capsule, turn off the dead end feature
                            # because the capsule might be in the dead end
                            if distToCapsule < lastDistToCapsule and self.isDeadEnd(bestCapsule) and self.isDeadEnd((x, y)) and ghostDistToCapsule > distToCapsule:
                                features['in-dead-end'] = 0
                                features['dist-to-ghost'] = -features['dist-to-ghost']
                else:
                    distToHome = self.getMazeDistance(self.startPos, myNewPos)
                    features['closest-border'] = float(distToHome) / (walls.width * walls.height)

            # This means our current belief on the ghost position is wrong
            if self.closestGhost and self.getMazeDistance(myNewPos, self.closestGhost.getPosition()) <= 5 and closestGhost is None:
                self.closestGhost = None

            if features['dist-to-ghost'] == 0.0:
                features['dist-to-ghost'] = float(6) / (walls.width * walls.height)

            if len(capsules) > 0:
                if myNewPos in capsules:
                    features['eats-capsule'] = 1.0

            if features['in-danger'] != 1:
                distToBorder = min([self.getMazeDistance(myNewPos, e) for e in self.entries])

                if movesLeft - 10 < distToBorder or len(foods.asList()) <= 2:
                    distToHome = self.getMazeDistance(self.startPos, myNewPos)
                    features['closest-border'] = float(distToHome) / (walls.width * walls.height)
                else:
                    maxScore = -999999
                    bestFood = None
                    if len(foods.asList()) > 0:
                        for f in foods.asList():
                            distToGhost = self.getMazeDistance(f, self.closestGhost.getPosition()) if self.closestGhost else 0
                            distToFood = self.getMazeDistance(myNewPos, f)
                            score = distToGhost - distToFood
                            if score > maxScore:
                                maxScore = score
                                bestFood = f

                    distToFood = self.getMazeDistance(myNewPos, bestFood) if bestFood is not None else None
                    # distToFood = min([self.getMazeDistance(myNewPos, f) for f in foods.asList()]) if len(foods.asList()) > 0 else None

                    if bestFood is not None and (distToFood < distToBorder or myNewState.numCarrying == 0) and myCurState.numCarrying < int(self.totalFoodNum / 2):
                        features['closest-food'] = float(distToFood) / (walls.width * walls.height)
                    else:
                        features['closest-border'] = float(distToBorder) / (walls.width * walls.height)
                    if foods[x][y]:
                        features['eats-food'] = 1.0

        features.divideAll(10)

        return features

    def calculateReward(self, gameState):
        reward = 0

        lastFoods = self.getFood(self.lastGameState)
        lastCapsules = self.getCapsules(self.lastGameState)

        myState = self.getMyState(gameState)
        myLastState = self.getMyState(self.lastGameState)
        myPos = self.getMyPosition(gameState)
        myLastPos = self.getMyPosition(self.lastGameState)
        x, y = myPos
        x = int(x)
        y = int(y)

        if myPos == myLastPos:
            reward -= 1

        if lastFoods[x][y]:
            reward += 1
        if myPos in lastCapsules:
            reward += 20

        if myPos == self.startPos and not myState.isPacman and myLastState.isPacman:
            reward -= 100

        if myLastState.numCarrying > myState.numCarrying and myPos != self.startPos:
            reward += 40 * myLastState.numCarrying

        return reward


class ReflexAndGRDefensiveAgent(CaptureAgent):
    def doAction(self, gameState, action):
        self.lastAction = action
        self.lastGameState = gameState

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        self.start = gameState.getAgentPosition(self.index)
        self.entries = []
        walls = gameState.getWalls()
        mapWidth = walls.width
        mapHeight = walls.height
        if self.red:
            borderX = int(mapWidth / 2) - 1
        else:
            borderX = int(mapWidth / 2)
        for borderY in range(0, mapHeight):
            if not walls[borderX][borderY]:
                self.entries.append((borderX, borderY))

        minDistToEntries = 10000000
        self.guardPoint = (0, 0)
        startX = 0 if self.red else borderX
        endX = borderX + 1 if self.red else mapWidth
        for x in range(startX, endX):
            for y in range(0, mapHeight):
                if not walls[x][y]:
                    dist_to_entries = sum(self.getMazeDistance((x, y), entry) for entry in self.entries)
                    if dist_to_entries < minDistToEntries:
                        minDistToEntries = dist_to_entries
                        self.guardPoint = (x, y)

        self.target = None
        self.lastGameState = None
        self.lastAction = None

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        selectedAction = random.choice(bestActions)
        self.doAction(gameState, selectedAction)
        return selectedAction

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else: # is it for option STOP?
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            self.target = None
            dist = min([self.getMazeDistance(myPos, a.getPosition()) for a in invaders])
            agentState = gameState.getAgentState(self.index)
            if dist <= 3 and agentState.scaredTimer > 0:
                features['invaderDistance'] = -dist
            else:
                features['invaderDistance'] = dist
        else:
            if self.lastGameState:
                lastFoods = self.getFoodYouAreDefending(self.lastGameState).asList()
                curFoods = self.getFoodYouAreDefending(gameState).asList()
                difference = set(lastFoods).difference(curFoods)

                if len(difference) > 0:
                    # Find the food that is closest to the missing food
                    missingFood = difference.pop()
                    nextFood = None
                    minDist = 1000000
                    for food in curFoods:
                        dist = self.getMazeDistance(food, missingFood)
                        if dist < minDist:
                            minDist = dist
                            nextFood = food
                    self.target = nextFood

            if self.target is None:
                dist = self.getMazeDistance(myPos, self.guardPoint)
                features['guardPointDistance'] = dist
            else:
                dist = self.getMazeDistance(myPos, self.target)
                features['targetDistance'] = dist

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1
        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -10,
            'stop': -100,
            'reverse': -2,
            'guardPointDistance': -10,
            'targetDistance': -10,
        }


class DefensiveAgentWithOffenseTactic(CaptureAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.attack = OffensiveAgent(*args, **kwargs)
        self.defense = ReflexAndGRDefensiveAgent(*args, **kwargs)

    def registerInitialState(self, gameState):
        self.attack.registerInitialState(gameState)
        self.defense.registerInitialState(gameState)

    def chooseAction(self, gameState):
        myState = gameState.getAgentState(self.index)

        if myState.scaredTimer > 0:
            return self.attack.chooseAction(gameState)
        else:
            return self.defense.chooseAction(gameState)
