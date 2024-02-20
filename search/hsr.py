import pathlib
import networkx as nx
import numpy as np
import pandas as pd
import util
import cProfile
import time
import sys


cwd = pathlib.Path(__file__).parent.resolve()


class HighSpeedRailProblem:
    """
    A search problem intended to explore what an optimal national HSR network
    could look like for the US, given various performance metrics and budgetary constraints.
    - The state space is all possible combinations of intercity rail segments.
    A rail segment is represented as a pair of two cities.
    - The action space is simply placing any rail segment that is not present.
    """

    def __init__(self, od_matrix: pd.DataFrame, colsForHash: list[str] = ['Origin', 'Dest']) -> None:
        # od_matrix is origin-destination matrix; DataFrame of all city pairs and their associated metrics
        self.od_matrix = od_matrix
        # hash is an order-independent identifier for a single OD pair: (JFK,LAX) == (LAX,JFK)
        self.od_matrix['hash'] = [frozenset(x) for x in zip(od_matrix['Origin'], od_matrix['Dest'])]
        self.colsForHash = colsForHash
        self.hashToCostMetrics: dict[frozenset, pd.DataFrame] = {}

    def getStartState(self):
        return util.HSRSearchState(pd.DataFrame(columns=self.od_matrix.columns))

    def isGoalState(self, state: util.HSRSearchState):
        pass

    def getValidActions(self, state: util.HSRSearchState):
        pass

    def getOnlySuccessors(self, state: util.HSRSearchState) -> list[util.HSRSearchState]:
        actions = self.getValidActions(state)
        nextStates: pd.Series = actions.apply(lambda action: state.getSuccessor(action), axis='columns')  # type: ignore
        return nextStates.tolist()

    def getCostOfState(self, state: util.HSRSearchState):
        paths_df = self.getAllRailPaths(state)
        path_attributes = self.getPathAttributes(paths_df, self.od_matrix)
        path_metrics = self.calcFinalMetrics(path_attributes)

        score_cols = [
            'delta_travel_time',
            'hsr_passengers',
            'flight_passengers',
            'total_passengers',
            'new_co2',
            'delta_co2',
        ]
        # NOTE: construction cost USD is not included in this dataframe because it would be overcounted
        #  -- many paths may share the same rail segments!
        # to get total construction cost, simply sum up cost of segments in state

        # todo normalization of metrics
        score_df = path_metrics[score_cols]
        sums = score_df.sum()

        self.hashToCostMetrics[state.getHash()] = score_df

        # 1000 kg co2 is weighted the same as 1 passenger
        cost = (sums['new_co2'] / 1e6) - sums['hsr_passengers']

        return cost

    def getAllRailPaths(self, state: util.HSRSearchState) -> pd.DataFrame:
        """
        Build a dataframe of shortest paths between every pair of cities in the provided rail network (state).

        Columns returned:
        - hash
        - Origin
        - Dest
        - path_origin
        - path_dest
        """
        # represent state as a graph of rail segments
        G = nx.from_pandas_edgelist(state.getRailSegments(), source='Origin', target='Dest', edge_attr='NonStopKm')

        # build dataframe of shortest paths between every pair of cities in the rail network
        # paths_df will contain one row for each leg of each shortest path
        paths_tuples: list[tuple[str, dict[str, list[str]]]] = nx.all_pairs_dijkstra_path(G, weight='NonStopKm')
        visited_paths: set[frozenset] = set()
        paths_df = pd.DataFrame(columns=['hash', 'Origin', 'Dest', 'path_origin', 'path_dest'])

        for (origin, paths_dict) in paths_tuples:
            for (dest, path) in paths_dict.items():
                pathKey = frozenset([origin, dest])
                # skip 0-length and already-seen paths
                # TODO skip paths between airports in same city
                if origin == dest or pathKey in visited_paths:
                    continue
                visited_paths.add(pathKey)

                # create small dataframe to represent this path; each row is an edge
                df = pd.DataFrame(path, columns=['Origin'])
                df['Dest'] = df['Origin'].shift(-1)
                df = df.dropna()
                df['path_origin'] = path[0]
                df['path_dest'] = path[-1]
                # hash is an order-independent identifier for a single edge: (JFK,LAX) == (LAX,JFK)
                df['hash'] = [frozenset(x) for x in zip(df['Origin'], df['Dest'])]
                paths_df = pd.concat([paths_df, df], axis='index')

        return paths_df

    def getPathAttributes(self, paths_df: pd.DataFrame, od_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Merge paths_df with the OD-matrix to obtain attributes for each rail segment present in the network.

        For each path, sum the following:
        - travel time by HSR
        - travel distance by HSR
        - co2 emitted by HSR journey

        For each path, count the following:
        - number of legs in the rail journey
        """

        # merge paths_df with the OD-matrix to obtain attributes for each rail segment present in the network
        paths_df = paths_df.merge(od_matrix.drop(columns=['Origin', 'Dest']), on='hash', how='left', validate='m:1')
        # group paths_df by unique path
        paths_df['path_hash'] = [frozenset(x) for x in zip(paths_df['path_origin'], paths_df['path_dest'])]
        path_groups = paths_df.groupby('path_hash')[[
            'NonStopKm',
            'hsr_time_hr',
            'co2_g_hsr',
        ]]

        # for each path, sum the following:
        # - travel time by HSR
        # - travel distance by HSR -> becomes hsr_km (distance by rail journey)
        # - co2 emitted by HSR journey
        hsr_metrics = path_groups.sum()
        # for each path, count the following:
        # - number of legs in the rail journey
        hsr_metrics['hsr_legs'] = path_groups.count()['NonStopKm']
        hsr_metrics = hsr_metrics.rename(columns={'NonStopKm': 'hsr_km'}).reset_index()

        # merge OD-matrix back into aggregated path data to link metrics for HSR journey/path with
        #  corresponding metrics for direct flight journey
        hsr_metrics = hsr_metrics.merge(
            od_matrix[[
                'hash',
                'Origin', 'Dest',
                'NonStopKm',
                'Passengers',
                'flight_time_hr',
                'co2_g_flight',
                # 'city_origin', 'state_origin', 'pop_origin',
                # 'city_dest', 'state_dest', 'pop_dest',
            ]],
            left_on='path_hash',
            right_on='hash',
            how='outer',
            validate='1:1',
        ).rename(columns={'Passengers': 'total_passengers', 'NonStopKm': 'flight_km'}).drop(columns=['hash'])

        return hsr_metrics

    def calcFinalMetrics(self, hsr_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the final metrics that will be used for the cost function.

        Metrics calculated:
        - delta_travel_time -- change in travel time when opting for HSR over flying on this route
        - hsr_passengers -- number of passengers who will opt to take HSR on this route instead of flying
        - flight_passengers -- number of passengers who will still opt to fly this route
        - total_passengers -- total annual passenger volume on this route
        - new_co2 -- total annual co2 emissions from combined HSR + flight volumes on this route (units: grams)
        - delta_co2 -- change in total annual co2 emissions on this route
        """

        metrics = hsr_metrics.copy()

        # add 3 hours for travel time to/from airport, security, gate times, etc.
        metrics['flight_time_hr'] += 3

        metrics['delta_travel_time'] = (metrics['hsr_time_hr'] - metrics['flight_time_hr']).fillna(0)

        # number of passengers who will opt to take this HSR path instead of the corresponding flight
        # assumed proportion of passengers who will opt to take HSR over flying:
        # let t = time(HSR) / time(flight):
        # prop0: t<1.0 = 1.0 = 100% of passengers
        # prop1: 1.00-1.25 = 0.9
        # prop2: 1.25-1.50 = 0.8
        # prop3: 1.50-2.0 = 0.7
        # prop4: t>2.0 = 0.5

        t = metrics['hsr_time_hr'] / metrics['flight_time_hr']
        prop0 = t > 0
        prop1 = t > 1
        prop2 = t > 1.25
        prop3 = t > 1.50
        prop4 = t > 2.0

        hsr_passengers = pd.Series(index=metrics['total_passengers'].index)
        hsr_passengers.loc[prop0] = metrics['total_passengers']
        hsr_passengers.loc[prop1] = metrics['total_passengers'] * 0.9
        hsr_passengers.loc[prop2] = metrics['total_passengers'] * 0.8
        hsr_passengers.loc[prop3] = metrics['total_passengers'] * 0.7
        hsr_passengers.loc[prop4] = metrics['total_passengers'] * 0.5
        metrics['hsr_passengers'] = hsr_passengers.fillna(0)

        metrics['flight_passengers'] = metrics['total_passengers'] - metrics['hsr_passengers']

        # source: https://travelandclimate.org/transport-calculations
        # pkm = passenger-kilometer
        # units: g/pkm
        co2_pkm_flight = 133  # from 'Scheduled flight (Economy)'
        co2_pkm_hsr = 24  # from 'Electric train (Europe)'
        # change in co2 when switching from flight -> HSR
        metrics['new_co2'] = (
            (co2_pkm_flight * metrics['flight_passengers'] * metrics['flight_km'])
            + (co2_pkm_hsr * metrics['hsr_passengers'] * metrics['hsr_km'])
        ).fillna(metrics['co2_g_flight'])

        metrics['delta_co2'] = metrics['new_co2'] - metrics['co2_g_flight']

        return metrics


class HSRProblem1(HighSpeedRailProblem):
    """
    Problem 1 is exploring the following question:
    Given a fixed construction budget, what is the optimal selection of intercity rail corridors to build?
    - The cost of building a certain rail segment is the negative of the "benefit"
    gained from the segment, as defined by the metrics we are interested in.
    - The goal state is any state in which the total monetary costs of constructing
    all present rail segments is equal to the predetermined budget.
    """

    def __init__(self, od_matrix: pd.DataFrame, colsForHash: list[str] = ['Origin', 'Dest'], budget: float = 0) -> None:
        super().__init__(od_matrix=od_matrix, colsForHash=colsForHash)
        self.budget = budget

    def isGoalState(self, state: util.HSRSearchState):
        totalConstructionCost = state.getRailSegments()['construction_cost_usd'].sum()
        return totalConstructionCost >= self.budget

    def getValidActions(self, state: util.HSRSearchState) -> pd.DataFrame:
        """
        Returns a DataFrame of valid "actions" that can be taken from the given state.
        An action is placing a new intercity rail segment, represented by a row in the returned DataFrame.
        """
        cols = self.od_matrix.columns
        merged = self.od_matrix.merge(
            state.getRailSegments(), how='left', indicator=True,
            on=['Origin', 'Dest'], suffixes=('', '_state')
        )
        mask = merged['_merge'] == 'left_only'
        actions = merged[mask]

        # valid actions are those that result in a state with construction cost within our budget
        currConstructionCost = state.getRailSegments()['construction_cost_usd'].sum()
        actions = actions[(actions['construction_cost_usd'] + currConstructionCost) <= self.budget]

        return actions[cols]


class HSRProblem2(HighSpeedRailProblem):
    """
    Problem 2 is exploring the following question:
    Given a goal of improving national metrics (emissions, population served by rail, etc.)
    by a specified amount, what is the least-cost investment necessary to achieve this,
    and which arrangement of intercity rail corridors would satisfy this?
    - The cost of building a certain rail segment is the construction cost.
    - The goal state is any state in which the total benefit gained by all rail segments
    is at least the predetermined threshold.
    """

    def __init__(self, od_matrix: pd.DataFrame, colsForHash: list[str] = ['Origin', 'Dest'], threshold: float = 0.5) -> None:
        super().__init__(od_matrix=od_matrix, colsForHash=colsForHash)
        self.threshold = threshold

    def isGoalState(self, state: util.HSRSearchState):
        _ = super().getCostOfState(state)  # todo only for calulating metrics
        score_df = self.hashToCostMetrics[state.getHash()]
        old_co2 = self.od_matrix['co2_g_flight'].sum()
        new_co2 = score_df['new_co2'].sum()
        return (new_co2 / old_co2) <= self.threshold

    def getValidActions(self, state: util.HSRSearchState) -> pd.DataFrame:
        """
        Returns a DataFrame of valid "actions" that can be taken from the given state.
        An action is placing a new intercity rail segment, represented by a row in the returned DataFrame.
        """
        cols = self.od_matrix.columns
        merged = self.od_matrix.merge(
            state.getRailSegments(), how='left', indicator=True,
            on=['Origin', 'Dest'], suffixes=('', '_state')
        )
        mask = merged['_merge'] == 'left_only'
        actions = merged[mask]
        return actions[cols]

    def getCostOfState(self, state: util.HSRSearchState) -> float:
        return state.getRailSegments()['construction_cost_usd'].sum()


def HSRSearch(problem: HighSpeedRailProblem, heuristic=util.nullHeuristic):
    # dict of state hashes to states
    hashToState: dict[frozenset, util.HSRSearchState] = {}
    # dict of state hashes to state costs (eliminates need for frontier to hold costs)
    hashToCost: dict[frozenset, float] = {}
    # explored is set of (hashed) states
    explored: set[frozenset] = set()

    # a state is a dataframe of all intercity rail segments
    # cost is defined by the cost function of the problem
    startState = problem.getStartState()
    startStateHash = startState.getHash()
    hashToState[startStateHash] = startState
    hashToCost[startStateHash] = 0

    frontier = util.PriorityQueue()
    frontier.push(startStateHash, 0)

    # used sys to replace terminal outputs instead of printing 3-5k print statements
    print("initiating...")
    while not frontier.isEmpty():
        sys.stdout.write("\rexplored: {}".format(len(explored)))
        sys.stdout.flush()

        currentStateHash = frontier.pop()
        currentState = hashToState[currentStateHash]
        # currentCost = hashToCost[currentStateHash] # todo needed?
        explored.add(currentStateHash)

        if problem.isGoalState(currentState):
            return currentState

        successors = problem.getOnlySuccessors(currentState)
        for nextState in successors:
            nextStateHash = nextState.getHash()
            if nextStateHash not in hashToState:
                hashToState[nextStateHash] = nextState
            # compute cost only if we've never encountered this state (hash) before
            if nextStateHash not in hashToCost:
                hashToCost[nextStateHash] = problem.getCostOfState(nextState)

            nextCost = hashToCost[nextStateHash]
            nextPriority = nextCost + heuristic(nextState, problem)

            if nextStateHash not in explored:
                frontier.update(nextStateHash, nextPriority)

    return False  # if frontier empty, no solution


def main():
    start = time.time()
    # to cut down on the search space, only consider cities with a minimum population
    MIN_POP = 1e6
    MAX_POP = 1e99
    BUDGET_USD = 10e9

    # todo: decide: each city is either represented by its name or its IATA airport code (BOS, LAX...)
    od_matrix = pd.read_csv(cwd.joinpath('../data/algo_testing_data.csv'))

    mask1 = (od_matrix['pop_origin'] >= MIN_POP) &\
        (od_matrix['pop_origin'] <= MAX_POP) &\
        (od_matrix['pop_dest'] >= MIN_POP) &\
        (od_matrix['pop_dest'] <= MAX_POP)
    filtered_od1 = od_matrix[mask1]

    mask2 = (od_matrix['NonStopKm'] >= 100)\
        & (od_matrix['NonStopKm'] <= 700)\
        & (od_matrix['Passengers'] >= 10e3)
    filtered_od2 = od_matrix[mask2].sort_values('Passengers', ascending=False).head(12)
    print('input OD-matrix size:', len(filtered_od2))

    hsr1 = HSRProblem1(od_matrix=filtered_od1, budget=BUDGET_USD)
    hsr2 = HSRProblem2(od_matrix=filtered_od2, threshold=0.25)

    # solution = HSRSearch(hsr1)
    solution = HSRSearch(hsr2)
    if solution:
        print('solution found! updating csv...')
        solution.railSegments.to_csv(cwd.joinpath('../out/solution.csv'), index=False)
    else:
        print('\nno solution found :(')

    end = time.time()
    print('Runtime in seconds:', end - start)


if __name__ == '__main__':
    # cProfile.run('main()', sort='cumulative')  # for time analysis
    main()
