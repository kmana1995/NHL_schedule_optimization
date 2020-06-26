import requests
import json
import pandas as pd
from math import *
import os
from pyomo.environ import *
import gmplot


# Get the 2019-2020 schedule
schedule = pd.read_csv(os.path.join(os.path.dirname(__file__), 'nhl_schedule_2019_2020.csv'))
schedule.columns = schedule.columns.str.upper()
schedule = schedule[['HOME', 'VISITOR']]
schedule['HOME'] = schedule['HOME'].str.strip()
schedule['VISITOR'] = schedule['VISITOR'].str.strip()

class NHLScheduleOptimizer:

    def __init__(self, team, max_games_per_trip=4):
        self.team = team
        self.max_games_per_trip = max_games_per_trip

    def optimize_schedule(self):
        """
        Wrapper function to be called to optimize schedule
        """
        # get the raw data
        arena_locations = self.get_arena_locations()
        full_nhl_schedule = self.get_schedule_data()

        # format the data in a dictionary to be used in the pyomo solver
        lp_data = self.create_lp_data(arena_locations, full_nhl_schedule)

        abstract_model = self.create_abstract_model()

        optimal_distance, route_solution_df = self.run_full_schedule_optimization(abstract_model, lp_data, arena_locations)

        schedule_distance = self.calculate_actual_schedule_travel(arena_locations, full_nhl_schedule)
        print('Actual schedule travel distance calculated at {} for the {}'.format(round(schedule_distance, 2), self.team))
        print({'Optimal distance found to be {} for the {}'.format(optimal_distance, self.team)})
        print('Optimal Route Schedules:')
        print(route_solution_df)
        route_solution_df.to_csv('schedule_optimized_solution.csv')

    @staticmethod
    def get_schedule_data():
        """
        Get the NHL schedule
        :return schedule:
        """
        # Get the 2019-2020 schedule
        schedule = pd.read_csv(os.path.join(os.path.dirname(__file__), 'nhl_schedule_2019_2020.csv'), parse_dates=['Date'])
        schedule.columns = schedule.columns.str.upper()
        schedule = schedule[['DATE','HOME','VISITOR']]
        schedule['HOME'] = schedule['HOME'].str.strip()
        schedule['VISITOR'] = schedule['VISITOR'].str.strip()

        return schedule

    def get_arena_locations(self, source='csv'):
        """
        Get arena locations either from csv or from json url

        :param source:
        :return:
        """
        if source == 'csv':
            arena_locations = pd.read_csv(os.path.join(os.path.dirname(__file__), 'arena_locations.csv'))
        else:
            arena_locations = self.get_arena_locations_from_json()
        return arena_locations

    def get_arena_locations_from_json(self):
        """
        Get the arena locations from a provided url

        :return:
        """
        # read the long/lats of every nhl location
        url = "https://raw.githubusercontent.com/nhlscorebot/arenas/master/teams.json"

        raw_json = requests.get(url).text
        json_dictionary = json.loads(raw_json)

        teams = ['Vegas Golden Knights']
        arenas = ['T-mobile Arena']
        latitude = [36.1029]
        longitude = [-115.17]

        for key in list(json_dictionary.keys()):
            if key == 'Phoenix Coyotes':
                n_key = 'Arizona Coyotes'
                teams.append(n_key)
            else:
                teams.append(str(key).strip())
            arenas.append(str(json_dictionary[key]["arena"]).strip())
            latitude.append(float(json_dictionary[key]["lat"]))
            longitude.append(float(json_dictionary[key]["long"]))

        dataframe = pd.DataFrame({'TEAM': teams, 'ARENA': arenas, 'LAT': latitude, 'LONG': longitude})
        dataframe.to_csv(os.path.join(os.path.dirname(__file__), 'arena_locations.csv'))

        return dataframe

    @staticmethod
    def calculate_haversine_distance(location_1, location_2):
        """
        Calculate the haversine distance (in miles) between two locations

        :param location_1: longitude and latitude of location 1
        :param location_2: longitude and latitude of location 2
        :return haversine: the haversine distance in miles between location 1 and 2
        """

        # convert decimals to radians
        long_1, long_2, lat_1, lat_2 = map(radians, [location_1['LONG'].item(), location_2['LONG'].item(),
                                          location_1['LAT'].item(), location_2['LAT'].item()])

        # calc the lat and long differences between locations
        longitude_dif = long_1 - long_2
        latitude_dif = lat_1 - lat_2

        # Calculate the haversine distance
        haversine = 2*asin(sqrt(sin(latitude_dif/2)**2+cos(lat_2)*cos(lat_1)*sin(longitude_dif/2)**2))
        haversine = haversine*3956
        return haversine

    def calculate_actual_schedule_travel(self, arena_locations, schedule):

        team_schedule = schedule[(schedule['VISITOR'] == self.team) | (schedule['HOME'] == self.team)]

        ordered_schedule = team_schedule.sort_values('DATE').reset_index(drop=True)

        total_distance = 0
        for ind, row in ordered_schedule[:-1].iterrows():
            from_location = row['HOME']
            from_location = arena_locations[arena_locations['TEAM'] == from_location]
            to_location = ordered_schedule['HOME'][ind+1]
            to_location = arena_locations[arena_locations['TEAM'] == to_location]

            distance = self.calculate_haversine_distance(from_location, to_location)

            total_distance += distance

        trip_home = arena_locations[arena_locations['TEAM'] == self.team]
        total_distance += self.calculate_haversine_distance(to_location, trip_home)

        return total_distance

    def create_lp_data(self, arena_locations, schedule):
        """
        Function to "parcel" the into a format consumable by the LP model

        :param arena_locations: df, the locations of each arena
        :param schedule: df, the team schedules (all teams)
        :return data: dictionary of data formatted to be ingested by LP
        """
        # clarify home
        home = self.team

        # subset the full nhl schedule on the given team
        away_schedule = schedule[schedule['VISITOR'] == self.team]

        # clarify number of games remaining
        games_remaining = len(away_schedule)

        # create a game location set, consisting of every location the team must visit (including home)
        game_location_set = list(away_schedule['HOME'].unique())
        game_location_set.append(self.team)

        # calculate the number of away locations, remove the home team
        n_locations = len(game_location_set)-1

        # arc set, consisting of possible too --> from combinations
        arc_set = []
        for team_a in game_location_set:
            for team_b in game_location_set:
                if team_a == team_b:
                    continue
                arc = (team_a, team_b)
                arc_set.append(arc)

        # create a distance for each arc (distance between arenas)
        distance_param = {}
        for team_a, team_b in arc_set:
            distance = self.calculate_haversine_distance(arena_locations[arena_locations['TEAM'] == team_a],
                                                    arena_locations[arena_locations['TEAM'] == team_b])
            distance_param.update({(team_a, team_b): distance})

        # Calculate the number of games each location
        n_games_per_location_param = {}
        for location in game_location_set:
            n_visits = int(len(away_schedule[away_schedule['HOME'] == location]))
            if location == self.team:
                n_visits = 1000
            n_games_per_location_param.update({location: n_visits})

        # Store data in a dictionary
        data = {None: {
            'home' : {None: [home]},
            'n_locations' : {None: [n_locations]},
            'games_remaining': {None: [games_remaining]},
            'game_location_set': {None: game_location_set},
            'arc_set': {None: arc_set},
            'distance_param': distance_param,
            'n_games_per_location_param': n_games_per_location_param
        }}

        return data

    @staticmethod
    def objective_function(model):
        """
        Objective statement for our lp problem, minimize the distance traveled
        """
        return sum(model.route_var[loc_1, loc_2]*model.distance_param[loc_1, loc_2] for loc_1, loc_2 in model.arc_set)

    @staticmethod
    def one_trip_out_constraint(model, home):
        """
        Set a constraint that a trip must depart from home
        """
        return sum(model.route_var[home, :]) == 1

    @staticmethod
    def one_trip_in_constraint(model, home):
        """
        Set a constraint that a trip must return home
        """
        return sum(model.route_var[:, home]) == 1

    @staticmethod
    def inbound_outbound_constraint(model, location):
        """
        Set a constraint that a trip must have an equal number of trips in and out at each location (including return home)
        """
        return sum(model.route_var[location, :]) - sum(model.route_var[:, location]) == 0

    @staticmethod
    def keep_it_moving_constraint(model, from_location, to_location):
        """
        Set a constraint that a location pair cannot share just each other as destinations on a trip,
        unless one of them is home. This ensures circular routes returning home.
        """
        return sum(model.route_var[from_location, :]) - (
                model.route_var[from_location, to_location] + model.route_var[to_location, from_location]) >= 0

    def max_road_trip_constraint(self, model, home, n_teams):
        """
        Set a condition that each road trip must have at least four games, or the number of games remaining
        """
        return sum(model.route_var[:, :]) - sum(model.route_var[:, home]) == min(self.max_games_per_trip, n_teams)

    @staticmethod
    def overvisit_schedule_constraint(model, team):
        """
        Ensure that the team can't over-visit a location
        """
        return sum(model.route_var[:, team]) - model.n_games_per_location_param[team] <= 0

    def create_abstract_model(self):
        """
        Creating the abstract model to be used in optimization

        :return model: defined abstract model
        """
        model = AbstractModel()

        # define the sets and params
        model.home = Set(dimen=1)
        model.games_remaining = Set(dimen=1)
        model.n_locations = Set(dimen=1)
        model.game_location_set = Set(dimen=1)
        model.arc_set = Set(dimen=2)
        model.n_games_per_location_param = Param(model.game_location_set, within=NonNegativeIntegers)
        model.distance_param = Param(model.arc_set, within=NonNegativeReals)

        # define routes
        model.route_var = Var(model.arc_set, within=NonNegativeIntegers)

        # define objective
        model.objective = Objective(rule=self.objective_function, sense=minimize)

        # define constraints
        model.one_trip_in_constraint = Constraint(model.home, rule=self.one_trip_in_constraint)
        model.one_trip_out_constraint = Constraint(model.home,  rule=self.one_trip_out_constraint)
        model.keep_it_moving_constraint = Constraint(model.arc_set, rule=self.keep_it_moving_constraint)
        model.inbound_outbound_constraint = Constraint(model.game_location_set, rule=self.inbound_outbound_constraint)
        model.overvisit_schedule_constraint = Constraint(model.game_location_set, rule=self.overvisit_schedule_constraint)
        model.max_roadtrip_constraint = Constraint(model.home, model.n_locations, rule=self.max_road_trip_constraint)

        return model

    def run_full_schedule_optimization(self, abstract_model, data, arena_locations):
        """
        Function to run the iterative optimzation, allowing the team to leave for n games per trip until their
        schedule is exhausted.

        :param abstract_model: model
        :param data: data formatted in dictionary
        :param arena_locations: df, containing the arena locations
        :return optimal_distance: the optimal distance traveled
        :return route_solution_df: the route solutions
        """

        # create containers to store results
        from_longs = []
        from_lats = []
        to_longs = []
        to_lats = []
        from_loc = []
        to_loc = []
        routes = []
        optimal_distance = 0

        # iterate optimized routes until the schedule is exhausted
        route = 0
        while data[None]['games_remaining'][None][0] > 1:
            route += 1

            model_instance = abstract_model.create_instance(data)

            solver = SolverFactory('glpk')

            solution = solver.solve(model_instance)

            # if the solution is optimal, we print and store results, else we have an issue
            if str(solution.solver.termination_condition) == 'optimal':
                # add the route distance to the total optimal
                optimal_distance += model_instance.objective.expr()
                for to_location, from_location in model_instance.arc_set:

                    # if the number of trips is more than 1, we ran the route and want to show/store results
                    n_trips = model_instance.route_var.get_values()[to_location, from_location]
                    if model_instance.route_var.get_values()[to_location, from_location] > 0:
                        print('Flight from {} to {} made {} times on trip {}'.format(to_location, from_location,
                                                                          model_instance.route_var.get_values()[to_location, from_location], route))
                        # If the to location was a return trip, we do not deprecate from the games remaining
                        if to_location != self.team:
                            data[None]['games_remaining'][None][0] -= n_trips
                        data[None]['n_games_per_location_param'][to_location] -= n_trips
                        # if we have no more trips to this destination, we can consolidate for future optimizations
                        if data[None]['n_games_per_location_param'][to_location] == 0:
                            data[None]['n_locations'][None][0] -= 1
                        routes.append(route)
                        from_loc.append(from_location)
                        to_loc.append(to_location)
                        from_longs.append(arena_locations.loc[arena_locations['TEAM'] == from_location, 'LONG'].item())
                        from_lats.append(arena_locations.loc[arena_locations['TEAM'] == from_location, 'LAT'].item())
                        to_longs.append(arena_locations.loc[arena_locations['TEAM'] == to_location, 'LONG'].item())
                        to_lats.append(arena_locations.loc[arena_locations['TEAM'] == to_location, 'LAT'].item())
            else:
                print('Problem unsolved, termination condition {}'.format(str(solution.solver.termination_condition)))
                print('Optimization forced to stop with {} games remaining'.format(data[None]['games_remaining'][None][0]))
                break
        # store results in a df from containers
        raw_solution_df = pd.DataFrame({'ROUTE': routes, 'FROM_LOCATION': from_loc, 'FROM_LONG': from_longs, 'FROM_LAT': from_lats,
                                    'TO_LOCATION': to_loc, 'TO_LONGS': to_longs, 'TO_LATS': to_lats})

        # reformat the results to be more interpretable
        routes = []
        n_stop = []
        location = []
        lat = []
        long = []

        for route in raw_solution_df['ROUTE'].unique():
            subset = raw_solution_df[raw_solution_df['ROUTE'] == route]
            l1 = self.team
            destination = 1
            routes.append(route)
            n_stop.append(destination)
            location.append(l1)
            lat.append(subset.loc[subset['FROM_LOCATION'] == l1, 'FROM_LAT'].item())
            long.append(subset.loc[subset['FROM_LOCATION'] == l1, 'FROM_LONG'].item())
            while destination < len(subset):
                l1 = subset.loc[subset['FROM_LOCATION'] == l1, 'TO_LOCATION'].item()
                destination += 1
                routes.append(route)
                n_stop.append(destination)
                location.append(l1)
                lat.append(subset.loc[subset['FROM_LOCATION'] == l1, 'FROM_LAT'].item())
                long.append(subset.loc[subset['FROM_LOCATION'] == l1, 'FROM_LONG'].item())
            l1 = self.team
            destination += 1
            routes.append(route)
            n_stop.append(destination)
            location.append(l1)
            lat.append(subset.loc[subset['FROM_LOCATION'] == l1, 'FROM_LAT'].item())
            long.append(subset.loc[subset['FROM_LOCATION'] == l1, 'FROM_LONG'].item())

        route_solution_df = pd.DataFrame({'ROUTE': routes, 'N_STOP': n_stop, 'LOCATION': location, 'LAT': lat, 'LONG': long})

        return optimal_distance, route_solution_df

if __name__ == '__main__':
    optimizer = NHLScheduleOptimizer('Vancouver Canucks')
    optimizer.optimize_schedule()