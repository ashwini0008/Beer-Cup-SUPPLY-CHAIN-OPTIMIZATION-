import gurobipy as gp
from gurobipy import GRB
import json
import os
from prettytable import PrettyTable  # Ensure PrettyTable is installed

class BeerCupOptimizer:
    """
    Optimizer for Beer Cup distribution across multiple ports and destinations.
    Handles constraints including port capacities, delays, and special conditions.
    """
    def __init__(self, paths: list):
        self.scale_factor = 50
        self.min_ports = 5
        self.max_batches = 100
        self.min_shipment_per_port = 1

        self.port_capacities = {
            "Port A": 2000 / self.scale_factor,
            "Port B": 3000 / self.scale_factor,
            "Port C": 5000 / self.scale_factor,
            "Port D": 3000 / self.scale_factor,
            "Port E": 4500 / self.scale_factor,
            "Port F": 4000 / self.scale_factor,
            "Port G": 2500 / self.scale_factor,
            "Port H": 3000 / self.scale_factor,
            "Port I": 2000 / self.scale_factor,
            "Port J": 1500 / self.scale_factor
        }

        self.port_delays = {
            "Port A": 0, "Port B": 0, "Port C": 2,
            "Port D": 0, "Port E": 0, "Port F": 0,
            "Port G": 0, "Port H": 0, "Port I": 0,
            "Port J": 3
        }

        self.paths = paths
        self.path_map = self.process_paths()
        self.port_modes = self.extract_port_modes()
        self.activation_cost_map = self.build_activation_cost_map()

        # Filter port_modes based on presence in activation_cost_map
        self.port_modes = {
            (port, mode) for (port, mode) in self.port_modes
            if (port, mode) in self.activation_cost_map
        }

        # Check minimum ports
        if len(self.port_modes) < self.min_ports:
            raise ValueError("Insufficient port-mode combinations to satisfy the minimum ports constraint.")
        
        # Initialize dictionary to hold z_vars for Port A paths (for additional misc. cost)
        self.z_vars_A = {}

    def process_paths(self) -> dict:
        path_map = {}
        for path in self.paths:
            try:
                key = (
                    path["Port"],
                    path["Destination"],
                    path["Mode"].lower(),
                    int(path["Type of Beercup"])
                )
                path_map[key] = path
            except (KeyError, ValueError) as e:
                print(f"Skipping path due to error: {e}")
                continue
        return path_map

    def extract_port_modes(self) -> set:
        return {(path["Port"], path["Mode"].lower()) for path in self.paths}

    def build_activation_cost_map(self) -> dict:
        activation_cost_map = {}
        for path in self.paths:
            key = (path["Port"], path["Mode"].lower())
            activation_cost = float(path.get("Activation Cost of", 0))
            activation_cost_map.setdefault(key, activation_cost)
        return activation_cost_map

    def create_optimization_model(self, demands: dict) -> gp.Model:
        if not self.path_map:
            print("No paths available to create the optimization model.")
            return None

        mdl = gp.Model(name="Beer_Cup_Distribution")
        scaled_demands = {k: v / self.scale_factor for k, v in demands.items()}

        # Define shipment (x) variables
        self.x_vars = {
            key: mdl.addVar(
                name=f"x_{key[0]}_{key[1]}_{key[2]}_{key[3]}",
                lb=0, vtype=GRB.INTEGER
            )
            for key in self.path_map
        }

        # Define activation (y) variables
        self.y_vars = {
            (port, mode): mdl.addVar(name=f"y_{port}_{mode}", vtype=GRB.BINARY)
            for (port, mode) in self.port_modes
        }

        # ------------------------------
        # Define z_vars_A for Port A paths to apply fixed miscellaneous costs
        # ------------------------------
        for key in self.x_vars:
            if key[0] == "Port A":
                # Create a binary variable for each Port A path
                self.z_vars_A[key] = mdl.addVar(
                    name=f"z_{key[0]}_{key[1]}_{key[2]}_{key[3]}",
                    vtype=GRB.BINARY
                )

        # Add constraints to link z_vars_A with x_vars and y_vars
        M = self.max_batches  # Large constant for linking constraints
        for key, z_var in self.z_vars_A.items():
            # If any shipment is made on this path, z_var = 1
            mdl.addConstr(
                self.x_vars[key] <= M * z_var,
                name=f"Link_x_z_upper_{key}"
            )
            # z_var cannot exceed activation of the port-mode
            mdl.addConstr(
                z_var <= self.y_vars[(key[0], key[2])],
                name=f"Link_z_y_{key}"
            )

        # Add standard constraints
        self.add_constraints(mdl, self.x_vars, self.y_vars, scaled_demands)

        # Calculate total cost
        total_cost = self.calculate_total_cost(mdl, self.x_vars, self.y_vars)
        mdl.setObjective(total_cost, GRB.MINIMIZE)

        return mdl

    def add_constraints(self, mdl, x_vars, y_vars, scaled_demands):
        self.add_activation_constraints(mdl, x_vars, y_vars)
        self.add_capacity_constraints(mdl, x_vars)
        self.add_minimum_ports_constraint(mdl, y_vars)
        self.add_demand_constraints(mdl, x_vars, scaled_demands)

    def add_activation_constraints(self, mdl, x_vars, y_vars):
        for (port, mode), shipments in self.group_shipments_by_port_mode(x_vars).items():
            # If port-mode is activated (y=1), then x can be up to max_batches
            for x_var in shipments:
                mdl.addConstr(
                    x_var <= self.max_batches * y_vars[(port, mode)],
                    name=f"Activation_{port}_{mode}"
                )
            # Minimum shipments if port-mode is active
            mdl.addConstr(
                gp.quicksum(shipments) >= self.min_shipment_per_port * y_vars[(port, mode)],
                name=f"MinShipment_{port}_{mode}"
            )

    def group_shipments_by_port_mode(self, x_vars):
        port_mode_shipments = {}
        for key, x_var in x_vars.items():
            port, _, mode, _ = key
            port_mode_shipments.setdefault((port, mode), []).append(x_var)
        return port_mode_shipments

    def add_capacity_constraints(self, mdl, x_vars):
        """
        Capacity Constraints:
         - Each batch represents 50 units (self.scale_factor)
         - Sum up all types of beercups shipped from each port
         - Compare total units against port capacity
        """
        for port, capacity in self.port_capacities.items():
            relevant_keys = [k for k in x_vars if k[0] == port]
            if relevant_keys:
                total_units = gp.quicksum(
                    self.scale_factor * x_vars[k]
                    for k in relevant_keys
                )
                mdl.addConstr(
                    total_units <= capacity * self.scale_factor,
                    name=f"Capacity_{port}"
                )
                
                # Print debugging info (optional)
                print(f"\nPort {port} capacity constraint:")
                print(f"Max capacity: {capacity * self.scale_factor} units")
                relevant_paths = [f"{k[1]}-Type{k[3]}" for k in relevant_keys]
                print(f"Summing shipments from paths: {relevant_paths}")

    def add_minimum_ports_constraint(self, mdl, y_vars):
        mdl.addConstr(
            gp.quicksum(y_vars.values()) >= self.min_ports,
            name="Minimum_Ports"
        )

    def add_demand_constraints(self, mdl, x_vars, scaled_demands):
        """
        Demand Constraints:
         - For each destination & beer type, sum of shipments must equal demand
         - Special handling: Port H shipments are 0.7 of a standard batch
        """
        for (dest, beer_type), demand in scaled_demands.items():
            relevant_keys = [k for k in x_vars if k[1] == dest and k[3] == beer_type]
            if relevant_keys:
                # If port == "Port H", each batch counts as only 0.7
                terms = [0.7 * x_vars[k] if k[0] == "Port H" else x_vars[k] for k in relevant_keys]
                mdl.addConstr(
                    gp.quicksum(terms) == demand,
                    name=f"Demand_{dest}_{beer_type}"
                )

    def calculate_total_cost(self, mdl, x_vars, y_vars):
        """
        Summation of:
         - Transport + Packaging + Activation + Penalty + Misc - Reward
        """
        penalty_cost = self.calculate_penalty_cost(mdl, x_vars, y_vars)
        reward_cost = self.calculate_reward_cost(mdl, x_vars, y_vars)
        transport_cost = self.calculate_transport_cost(x_vars)
        packaging_cost = self.calculate_packaging_cost(x_vars)
        
        # Activation cost now depends on w_vars (defined in reward cost block),
        # so we pass 'mdl' as well and fetch them from the model:
        activation_cost = self.calculate_activation_cost(mdl, x_vars)

        misc_cost = self.calculate_miscellaneous_costs(x_vars)

        total_cost = (
            transport_cost +
            packaging_cost +
            activation_cost +
            penalty_cost +
            misc_cost -
            reward_cost
        )
        return total_cost

    def calculate_penalty_cost(self, mdl, x_vars, y_vars):
        """
        Penalty Cost:
         - If port != 'Port F': linear penalty
         - If port == 'Port F': use a separate set of variables for quadratic penalty
        """
        p_linear_vars = {}
        z_linear_vars = {}
        M = 1e6

        for k in x_vars:
            if k[0] != "Port F":
                p_linear_vars[k] = mdl.addVar(
                    name=f"p_linear_{k[0]}_{k[1]}_{k[2]}_{k[3]}",
                    lb=0, vtype=GRB.CONTINUOUS
                )
                z_linear_vars[k] = mdl.addVar(
                    name=f"z_linear_{k[0]}_{k[1]}_{k[2]}_{k[3]}",
                    vtype=GRB.BINARY
                )
                expression = (
                    self.port_delays[k[0]] +
                    self.path_map[k].get("Penalty Days without considering delay", 0)
                )

                mdl.addConstr(
                    p_linear_vars[k] >= expression,
                    name=f"p_linear_ge_expr_{k}"
                )
                mdl.addConstr(
                    p_linear_vars[k] >= 0,
                    name=f"p_linear_ge_zero_{k}"
                )
                mdl.addConstr(
                    p_linear_vars[k] <= expression + M * (1 - z_linear_vars[k]),
                    name=f"p_linear_le_expr_{k}"
                )
                mdl.addConstr(
                    p_linear_vars[k] <= M * z_linear_vars[k],
                    name=f"p_linear_le_M_{k}"
                )

        penalty_cost_linear = gp.quicksum(
            self.path_map[k].get("Penalty for Delay per day per batch", 0) *
            p_linear_vars[k] *
            x_vars[k]
            for k in x_vars if k[0] != "Port F"
        )

        p_quadratic_vars = {}
        z_quadratic_vars = {}
        w_quadratic_vars = {}

        for k in x_vars:
            if k[0] == "Port F":
                p_quadratic_vars[k] = mdl.addVar(
                    name=f"p_quadratic_{k[0]}_{k[1]}_{k[2]}_{k[3]}",
                    lb=0, vtype=GRB.CONTINUOUS
                )
                z_quadratic_vars[k] = mdl.addVar(
                    name=f"z_quadratic_{k[0]}_{k[1]}_{k[2]}_{k[3]}",
                    vtype=GRB.BINARY
                )
                w_quadratic_vars[k] = mdl.addVar(
                    name=f"w_quadratic_{k[0]}_{k[1]}_{k[2]}_{k[3]}",
                    lb=0, vtype=GRB.CONTINUOUS
                )

                expression = (
                    self.path_map[k].get("Penalty Days without considering delay", 0) +
                    0.5 * x_vars[k]
                )
                mdl.addConstr(
                    p_quadratic_vars[k] >= expression,
                    name=f"p_quadratic_ge_expr_{k}"
                )
                mdl.addConstr(
                    p_quadratic_vars[k] >= 0,
                    name=f"p_quadratic_ge_zero_{k}"
                )
                mdl.addConstr(
                    p_quadratic_vars[k] <= expression + M * (1 - z_quadratic_vars[k]),
                    name=f"p_quadratic_le_expr_{k}"
                )
                mdl.addConstr(
                    p_quadratic_vars[k] <= M * z_quadratic_vars[k],
                    name=f"p_quadratic_le_M_{k}"
                )

                # Link with y_vars
                mdl.addConstr(
                    w_quadratic_vars[k] <= p_quadratic_vars[k],
                    name=f"w_quadratic_le_p_{k}"
                )
                mdl.addConstr(
                    w_quadratic_vars[k] <= M * y_vars[(k[0], k[2])],
                    name=f"w_quadratic_le_My_{k}"
                )
                mdl.addConstr(
                    w_quadratic_vars[k] >= p_quadratic_vars[k] - M * (1 - y_vars[(k[0], k[2])]),
                    name=f"w_quadratic_ge_p_minus_My_{k}"
                )
                mdl.addConstr(
                    w_quadratic_vars[k] >= 0,
                    name=f"w_quadratic_ge_zero_{k}"
                )

        penalty_cost_quadratic = gp.quicksum(
            self.path_map[k].get("Penalty for Delay per day per batch", 0) *
            w_quadratic_vars[k] *
            x_vars[k]
            for k in x_vars if k[0] == "Port F"
        )

        return penalty_cost_linear + penalty_cost_quadratic
    
    def calculate_reward_cost(self, mdl, x_vars, y_vars):
        """
        Calculates the Reward Cost using:
         Reward = sum_k( Urgency_Bonus_k * max(0, RewardDays_k - Delay - (0.5*x_vars[k] if Port F)) ) * (binary usage)
        Implemented with auxiliary variables r_vars/binary_vars/w_vars.
        
        NOTE: We'll store self.w_vars so we can also use them in the activation cost.
        """
        r_vars = {}
        binary_vars = {}
        w_vars = {}
        M = 1e6

        for k in x_vars:
            var_base = f"{k[0]}_{k[1]}_{k[2]}_{k[3]}"
            r_vars[k] = mdl.addVar(
                name=f"r_{var_base}",
                lb=0, vtype=GRB.CONTINUOUS
            )
            binary_vars[k] = mdl.addVar(
                name=f"b_{var_base}",
                vtype=GRB.BINARY
            )
            # w_vars is used as a route-usage indicator: if x_vars[k] > 0 => w_vars[k] = 1
            w_vars[k] = mdl.addVar(
                name=f"w_{var_base}",
                vtype=GRB.BINARY
            )

        for k in x_vars:
            reward_days = float(self.path_map[k].get("Reward Days without considering delay", 0))
            port_delay = float(self.port_delays.get(k[0], 0))

            if k[0] == "Port F":
                expression = reward_days - port_delay - 0.5 * x_vars[k]
            else:
                expression = reward_days - port_delay

            # max(0, expression) constraints
            mdl.addConstr(
                r_vars[k] >= expression,
                name=f"r_ge_expr_{k}"
            )
            mdl.addConstr(
                r_vars[k] >= 0,
                name=f"r_ge_zero_{k}"
            )
            mdl.addConstr(
                r_vars[k] <= expression + M * (1 - binary_vars[k]),
                name=f"r_le_expr_plus_M_{k}"
            )
            mdl.addConstr(
                r_vars[k] <= M * binary_vars[k],
                name=f"r_le_M_{k}"
            )

            # Link w_vars to x_vars
            mdl.addConstr(
                w_vars[k] <= x_vars[k],
                name=f"w_le_x_{k}"
            )
            mdl.addConstr(
                x_vars[k] <= M * w_vars[k],
                name=f"x_le_Mw_{k}"
            )

        # Summation of all w_vars * UrgencyBonus
        reward_cost = gp.quicksum(
            float(self.path_map[k].get("Urgency Bonus per Day", 0)) * r_vars[k] * w_vars[k]
            for k in x_vars
        )

        # Store the w_vars so other methods can access them (e.g. for activation cost).
        self.w_vars = w_vars
        return reward_cost

    def calculate_transport_cost(self, x_vars):
        return gp.quicksum(
            float(self.path_map[k].get("Transportation Cost", 0)) * x_vars[k]
            for k in x_vars
        )

    def calculate_packaging_cost(self, x_vars):
        return gp.quicksum(
            float(self.path_map[k].get("Packaging Cost/unit of Beercup type", 0)) *
            self.scale_factor * x_vars[k]
            for k in x_vars
        )

    def calculate_activation_cost(self, mdl, x_vars):
        """
        Now activation cost is charged per route if that route is used:
        ActivationCost = Sum_over_routes [ (Activation Cost of route) * w_vars[route] ].
        """
        return gp.quicksum(
            float(self.path_map[k].get("Activation Cost of", 0)) * self.w_vars[k]
            for k in x_vars
        )

    def calculate_miscellaneous_costs(self, x_vars):
        """
        Miscellaneous Costs:
         - Port G has a fixed $4000 * 0.5 per batch
         - Port E adds 15% over (transport + packaging)
         - Port B & Port I have fixed $15 * scale_factor & $20 * scale_factor costs
         - Port A: additional fixed $10000 if z_vars_A = 1
        """
        misc_cost = gp.quicksum(
            4000 * 0.5 * x_vars[k]
            for k in x_vars if k[0] == "Port G"
        )
        misc_cost += gp.quicksum(
            0.15 * (
                float(self.path_map[k].get("Transportation Cost", 0)) +
                float(self.path_map[k].get("Packaging Cost/unit of Beercup type", 0)) * self.scale_factor
            ) * x_vars[k]
            for k in x_vars if k[0] == "Port E"
        )
        misc_cost += gp.quicksum(
            15 * self.scale_factor * x_vars[k]
            for k in x_vars if k[0] == "Port B"
        )
        misc_cost += gp.quicksum(
            20 * self.scale_factor * x_vars[k]
            for k in x_vars if k[0] == "Port I"
        )
        # Fixed cost for Port A if z_vars_A is active
        misc_cost += gp.quicksum(
            10000 * self.z_vars_A[k]
            for k in self.z_vars_A
        )

        return misc_cost

    def print_cost_components_during_optimization(self) -> None:
        """
        Prints the cost components expressions and their descriptions before optimization.
        """
        cost_table = PrettyTable()
        cost_table.field_names = ["Cost Component", "Expression"]

        # Define expressions as descriptions
        cost_table.add_row(["Transport Cost", "Sum of (Transportation Cost * x_vars)"])
        cost_table.add_row(["Packaging Cost", "Sum of (Packaging Cost/unit * scale_factor * x_vars)"])
        cost_table.add_row(["Activation Cost", "Sum of (Activation Cost of route * w_vars)"])
        cost_table.add_row(["Penalty Cost", "Sum of (Penalty Cost * penalty_vars * x_vars)"])
        cost_table.add_row(["Reward Cost", "Sum of (Urgency Bonus * r_vars * w_vars)"])
        cost_table.add_row(["Miscellaneous Cost", "Sum of (Various Miscellaneous Costs * x_vars)"])

        print("\nObjective Function Cost Components During Optimization:")
        print(cost_table)

    def print_cost_components_after_optimization(self, mdl: gp.Model) -> None:
        """
        Prints the numerical values of the cost components after optimization.
        """
        cost_table = PrettyTable()
        cost_table.field_names = ["Cost Component", "Amount ($)"]

        # Transport Cost
        transport_cost = sum(
            float(self.path_map[k].get("Transportation Cost", 0)) * self.x_vars[k].X
            for k in self.x_vars
        )

        # Packaging Cost
        packaging_cost = sum(
            float(self.path_map[k].get("Packaging Cost/unit of Beercup type", 0)) *
            self.scale_factor * self.x_vars[k].X
            for k in self.x_vars
        )

        # Activation Cost (via w_vars)
        activation_cost = sum(
            float(self.path_map[k].get("Activation Cost of", 0)) *
            mdl.getVarByName(f"w_{k[0]}_{k[1]}_{k[2]}_{k[3]}").X
            for k in self.x_vars
        )

        # Penalty Cost
        penalty_cost_linear = sum(
            float(self.path_map[k].get("Penalty for Delay per day per batch", 0)) *
            mdl.getVarByName(f"p_linear_{k[0]}_{k[1]}_{k[2]}_{k[3]}").X *
            self.x_vars[k].X
            for k in self.x_vars if k[0] != "Port F"
        )
        penalty_cost_quadratic = sum(
            float(self.path_map[k].get("Penalty for Delay per day per batch", 0)) *
            mdl.getVarByName(f"w_quadratic_{k[0]}_{k[1]}_{k[2]}_{k[3]}").X *
            self.x_vars[k].X
            for k in self.x_vars if k[0] == "Port F"
        )
        penalty_cost = penalty_cost_linear + penalty_cost_quadratic

        # Reward Cost
        reward_cost = 0
        for k in self.x_vars:
            w_var = mdl.getVarByName(f"w_{k[0]}_{k[1]}_{k[2]}_{k[3]}")
            r_var = mdl.getVarByName(f"r_{k[0]}_{k[1]}_{k[2]}_{k[3]}")
            if w_var and w_var.X > 0 and r_var:
                reward_cost += float(self.path_map[k].get("Urgency Bonus per Day", 0)) * r_var.X

        # Miscellaneous Cost
        misc_cost = sum(
            4000 * 0.5 * self.x_vars[k].X
            for k in self.x_vars if k[0] == "Port G"
        )
        misc_cost += sum(
            0.15 * (
                float(self.path_map[k].get("Transportation Cost", 0)) +
                float(self.path_map[k].get("Packaging Cost/unit of Beercup type", 0)) * self.scale_factor
            ) * self.x_vars[k].X
            for k in self.x_vars if k[0] == "Port E"
        )
        misc_cost += sum(
            15 * self.scale_factor * self.x_vars[k].X
            for k in self.x_vars if k[0] == "Port B"
        )
        misc_cost += sum(
            20 * self.scale_factor * self.x_vars[k].X
            for k in self.x_vars if k[0] == "Port I"
        )
        misc_cost += sum(
            10000 * self.z_vars_A[k].X
            for k in self.z_vars_A
        )

        # Populate the cost table
        cost_table.add_row(["Transport Cost", f"${transport_cost:,.2f}"])
        cost_table.add_row(["Packaging Cost", f"${packaging_cost:,.2f}"])
        cost_table.add_row(["Activation Cost", f"${activation_cost:,.2f}"])
        cost_table.add_row(["Penalty Cost", f"${penalty_cost:,.2f}"])
        cost_table.add_row(["Reward Cost", f"-${reward_cost:,.2f}"])
        cost_table.add_row(["Miscellaneous Cost", f"${misc_cost:,.2f}"])

        # Sum of Components
        sum_of_components = (
            transport_cost +
            packaging_cost +
            activation_cost +
            penalty_cost +
            misc_cost -
            reward_cost
        )
        cost_table.add_row(["Sum of Components", f"${sum_of_components:,.2f}"])

        # Total Objective Value
        total_obj_value = mdl.ObjVal
        cost_table.add_row(["Total Objective Value", f"${total_obj_value:,.2f}"])

        # Sum of All Positive Costs
        sum_all_positive_costs = (
            transport_cost +
            packaging_cost +
            activation_cost +
            penalty_cost +
            misc_cost
        )
        cost_table.add_row(["Sum of All Positive Costs", f"${sum_all_positive_costs:,.2f}"])

        print("\nObjective Function Cost Components After Optimization:")
        print(cost_table)

        # If there's a discrepancy, note it
        if abs(sum_of_components - total_obj_value) > 1:
            print("\nNote: The difference between sum of components and objective value "
                  "is due to the optimization model's handling of non-negative penalty constraints.")

    def solve_and_report(self, mdl: gp.Model) -> None:
        if mdl is None:
            print("No optimization model to solve.")
            return

        # Print cost components expressions before optimization
        self.print_cost_components_during_optimization()

        try:
            mdl.optimize()
        except gp.GurobiError as e:
            print(f"Gurobi Error: {e}")
            return
        except Exception as e:
            print(f"Unexpected Error: {e}")
            return

        if mdl.status == GRB.OPTIMAL:
            # After optimization, print cost components with numerical values
            self.print_cost_components_after_optimization(mdl)

            # Print solution summary
            self.print_solution_summary(mdl)

            # Collect solution details
            cost_components = {
                'transport_cost': 0.0,
                'packaging_cost': 0.0,
                'activation_cost': 0.0,
                'penalty_cost': 0.0,
                'reward_cost': 0.0,
                'misc_cost': 0.0
            }

            results, active_activation, total_batches, beercups_per_destination, quantities_per_port = \
                self.collect_solution_details(mdl, cost_components)

            # Print detailed reports
            self.print_detailed_reports(
                results,
                cost_components,
                mdl.ObjVal,
                active_activation,
                total_batches,
                beercups_per_destination,
                quantities_per_port
            )
        else:
            print("No feasible solution found!")

    def print_solution_summary(self, mdl: gp.Model) -> None:
        print(f"\nSolution status: {mdl.status}")
        if mdl.status == GRB.OPTIMAL:
            print(f"Objective Function Value: ${mdl.ObjVal:,.2f}")
        else:
            print("No optimal solution found.")

        active_ports_modes = [
            (key[0], key[1])
            for key in self.port_modes
            if self.y_vars[key].X > 0.5
        ]

        if active_ports_modes:
            print("\nActive Ports and Modes:")
            for port, mode in active_ports_modes:
                print(f"- {port} ({mode.capitalize()})")
        else:
            print("\nNo active ports and modes found in the solution.")

    def collect_solution_details(self, mdl: gp.Model, cost_components: dict):
        """
        Extracts route usage (x > 0) and aggregates cost info for final reporting.
        """
        results = []
        active_activation = {}
        total_batches = 0
        beercups_per_destination = {}
        quantities_per_port = {}

        for key, x_var in self.x_vars.items():
            if x_var.X > 0.001:
                route_details = self.analyze_route(key, x_var.X, mdl, cost_components)
                if route_details:
                    results.append(route_details)
                    total_batches += route_details["Number of Batches"]

                    dest = key[1]
                    beer_type = key[3]
                    beercups = route_details["Total Units"]

                    if dest not in beercups_per_destination:
                        beercups_per_destination[dest] = {99:0, 95:0, 90:0}
                    beercups_per_destination[dest][beer_type] += beercups

                    port = key[0]
                    if port not in quantities_per_port:
                        quantities_per_port[port] = 0
                    quantities_per_port[port] += beercups

        # Store the per-port-mode activation cost in active_activation, but 
        # now that we charge activation cost per route, this might just be 0 or unused.
        # We'll keep the dictionary for consistency, but won't print it in a separate table.
        for (port, mode), y_var in self.y_vars.items():
            if y_var.X > 0.5:
                # We do not double-count route-level activation
                # but we track the value for info if needed
                active_activation[(port, mode)] = self.activation_cost_map.get((port, mode), 0)

        return results, active_activation, total_batches, beercups_per_destination, quantities_per_port

    def analyze_route(self, key: tuple, x_value: float, mdl: gp.Model, cost_components: dict) -> dict:
        """
        Analyze route usage: calculate costs for a single route (port->destination).
        """
        try:
            port, dest, mode, beer_type = key
            path = self.path_map[key]
        except KeyError:
            return {}

        num_batches = x_value
        total_units = num_batches * self.scale_factor

        transport_cost = float(path.get("Transportation Cost", 0)) * num_batches
        packaging_cost = float(path.get("Packaging Cost/unit of Beercup type", 0)) * self.scale_factor * num_batches

        # Calculate penalty cost
        if port != "Port F":
            p_var_name = f"p_linear_{port}_{dest}_{mode}_{beer_type}"
            p_var = mdl.getVarByName(p_var_name)
            if p_var:
                penalty_days = max(0, p_var.X)
                penalty_cost = float(path.get("Penalty for Delay per day per batch", 0)) * penalty_days * num_batches
            else:
                penalty_cost = 0
        else:
            p_var_name = f"w_quadratic_{port}_{dest}_{mode}_{beer_type}"
            p_var = mdl.getVarByName(p_var_name)
            if p_var:
                penalty_days = max(0, p_var.X)
                penalty_cost = float(path.get("Penalty for Delay per day per batch", 0)) * penalty_days * num_batches
            else:
                penalty_cost = 0

        # Calculate reward cost
        reward_cost = 0
        w_var_name = f"w_{port}_{dest}_{mode}_{beer_type}"
        r_var_name = f"r_{port}_{dest}_{mode}_{beer_type}"

        w_var = mdl.getVarByName(w_var_name)
        r_var = mdl.getVarByName(r_var_name)
        
        if w_var and w_var.X > 0 and r_var:
            reward_cost = float(path.get("Urgency Bonus per Day", 0)) * r_var.X

        # Activation cost for this route
        if w_var and w_var.X > 0:
            activation_cost = float(path.get("Activation Cost of", 0))
        else:
            activation_cost = 0

        # Calculate available reward days for display
        reward_days = float(path.get("Reward Days without considering delay", 0))
        port_delay = float(self.port_delays.get(port, 0))
        if port == "Port F":
            additional_delay = 0.5 * num_batches
        else:
            additional_delay = 0
        available_reward_days = max(0, reward_days - port_delay - additional_delay)

        # Calculate penalty days for display
        base_penalty_days = float(path.get("Penalty Days without considering delay", 0))
        total_penalty_days = max(0, base_penalty_days + port_delay + additional_delay)

        # Calculate miscellaneous costs
        misc_cost = 0.0
        if port == "Port G":
            misc_cost += 4000 * 0.5 * num_batches
        if port == "Port E":
            misc_cost += 0.15 * (transport_cost + packaging_cost)
        if port == "Port B":
            misc_cost += 15 * self.scale_factor * num_batches
        if port == "Port I":
            misc_cost += 20 * self.scale_factor * num_batches
        if port == "Port A":
            z_var = self.z_vars_A.get(key)
            if z_var and z_var.X > 0.5:
                misc_cost += 10000

        # Update cost components
        cost_components['transport_cost'] += transport_cost
        cost_components['packaging_cost'] += packaging_cost
        cost_components['penalty_cost'] += penalty_cost
        cost_components['reward_cost'] += reward_cost
        cost_components['misc_cost'] += misc_cost
        cost_components['activation_cost'] += activation_cost

        return {
            "Port -> Destination(Mode)": f"{port} -> {dest} ({mode.capitalize()})",
            "Beercup Type": beer_type,
            "Number of Batches": num_batches,
            "Total Units": total_units,
            "Available Reward Days": available_reward_days,
            "Reward Cost": reward_cost,
            "Total Penalty Days": total_penalty_days,
            "Penalty Cost": penalty_cost,
            "Transport Cost": transport_cost,
            "Packaging Cost": packaging_cost,
            "Miscellaneous Cost": misc_cost,
            "Activation Cost": activation_cost
        }

    def print_detailed_reports(
        self,
        results: list,
        cost_components: dict,
        obj_value: float,
        active_activation: dict,
        total_batches: int,
        beercups_per_destination: dict,
        quantities_per_port: dict
    ) -> None:
        """
        Print comprehensive reports including:
         - Detailed/Basic Optimal Routes
         - (Removed the old Activation Costs table)
         - Beercup distribution
         - Summaries of cost components
        """
        if not results:
            print("No routes to display.")
            return

        # Detailed table
        detailed_table = PrettyTable()
        detailed_table.field_names = [
            "Port->Destination(Mode)",
            "Type",
            "Batches",
            "TotalUnits",
            "AvailableRewardDays",
            "RewardCost",
            "TotalPenaltyDays",
            "PenaltyCost",
            "TransportCost",
            "PackagingCost",
            "MiscellaneousCost",
            "ActivationCost"
        ]

        # Initialize sum variables for the detailed table
        sum_batches = 0.0
        sum_total_units = 0.0
        sum_available_reward_days = 0.0
        sum_reward_cost = 0.0
        sum_total_penalty_days = 0.0
        sum_penalty_cost = 0.0
        sum_transport_cost = 0.0
        sum_packaging_cost = 0.0
        sum_miscellaneous_cost = 0.0
        sum_activation_cost = 0.0

        # Basic table
        basic_table = PrettyTable()
        basic_table.field_names = [
            "Port->Destination(Mode)",
            "BeercupType",
            "NumberofBatches"
        ]

        # Table: Sum of Beercups per Destination
        beercups_table = PrettyTable()
        beercups_table.field_names = ["Destination", "Beercup Type 99", "Beercup Type 95", "Beercup Type 90"]

        # Table: Total Quantities Shipped from Each Port
        quantities_table = PrettyTable()
        quantities_table.field_names = ["Port", "Total Units Shipped"]

        # Populate the detailed/basic tables
        for res in results:
            detailed_table.add_row([
                res["Port -> Destination(Mode)"],
                res["Beercup Type"],
                res["Number of Batches"],
                res["Total Units"],
                f"{res['Available Reward Days']:.2f}",
                f"${res['Reward Cost']:.2f}",
                f"{res['Total Penalty Days']:.2f}",
                f"${res['Penalty Cost']:.2f}",
                f"${res['Transport Cost']:.2f}",
                f"${res['Packaging Cost']:.2f}",
                f"${res['Miscellaneous Cost']:.2f}",
                f"${res['Activation Cost']:.2f}"
            ])
            basic_table.add_row([
                res["Port -> Destination(Mode)"],
                res["Beercup Type"],
                res["Number of Batches"]
            ])

            # Accumulate sums
            sum_batches += res["Number of Batches"]
            sum_total_units += res["Total Units"]
            sum_available_reward_days += res["Available Reward Days"]
            sum_reward_cost += res["Reward Cost"]
            sum_total_penalty_days += res["Total Penalty Days"]
            sum_penalty_cost += res["Penalty Cost"]
            sum_transport_cost += res["Transport Cost"]
            sum_packaging_cost += res["Packaging Cost"]
            sum_miscellaneous_cost += res["Miscellaneous Cost"]
            sum_activation_cost += res["Activation Cost"]

        # Add the sum row to the detailed_table
        detailed_table.add_row([
            "Total",
            "",
            f"{sum_batches:.2f}",
            f"{sum_total_units:.2f}",
            f"{sum_available_reward_days:.2f}",
            f"${sum_reward_cost:.2f}",
            f"{sum_total_penalty_days:.2f}",
            f"${sum_penalty_cost:.2f}",
            f"${sum_transport_cost:.2f}",
            f"${sum_packaging_cost:.2f}",
            f"${sum_miscellaneous_cost:.2f}",
            f"${sum_activation_cost:.2f}"
        ])

        # Populate beercups_table
        for dest, types in beercups_per_destination.items():
            beercups_table.add_row([
                dest,
                types.get(99, 0),
                types.get(95, 0),
                types.get(90, 0)
            ])

        # Populate quantities_table
        for port, total_units in quantities_per_port.items():
            quantities_table.add_row([
                port,
                f"{int(total_units)}"  # convert float to int for neat display
            ])

        # Print tables
        print("\nDetailed Optimal Routes:")
        print(detailed_table)

        print("\nBasic Optimal Routes:")
        print(basic_table)

        print("\nTotal Number of Batches Shipped from All Ports:")
        print(f"{total_batches}")

        print("\nSum of Individual Types of Beercups Reached at Every Destination:")
        print(beercups_table)

        print("\nTotal Quantities Shipped from All Ports (Irrespective of Beercup Type):")
        print(quantities_table)



        # --------------------------
        # Cost Components Table
        # --------------------------
        cost_table = PrettyTable()
        cost_table.field_names = ["Cost Component", "Amount ($)"]

        transport_cost = cost_components['transport_cost']
        packaging_cost = cost_components['packaging_cost']
        activation_cost = cost_components['activation_cost']
        penalty_cost = max(0, cost_components['penalty_cost'])
        reward_cost = cost_components['reward_cost']
        misc_cost = cost_components['misc_cost']

        cost_table.add_row(["Transport Cost", f"${transport_cost:,.2f}"])
        cost_table.add_row(["Packaging Cost", f"${packaging_cost:,.2f}"])
        cost_table.add_row(["Activation Cost", f"${activation_cost:,.2f}"])
        cost_table.add_row(["Penalty Cost", f"${penalty_cost:,.2f}"])
        cost_table.add_row(["Reward Cost", f"-${reward_cost:,.2f}"])
        cost_table.add_row(["Miscellaneous Cost", f"${misc_cost:,.2f}"])

        # Sum of Components
        sum_of_components = (
            transport_cost +
            packaging_cost +
            activation_cost +
            penalty_cost +
            misc_cost -
            reward_cost
        )
        cost_table.add_row(["Sum of Components", f"${sum_of_components:,.2f}"])

        # Total Objective Value
        total_obj_value = obj_value
        cost_table.add_row(["Total Objective Value", f"${total_obj_value:,.2f}"])

        # Sum of All Positive Costs
        sum_all_positive_costs = (
            transport_cost +
            packaging_cost +
            activation_cost +
            penalty_cost +
            misc_cost
        )
        cost_table.add_row(["Sum of All Positive Costs", f"${sum_all_positive_costs:,.2f}"])

        print("\nCost Components Used in Objective Function:")
        print(cost_table)

        # If there's a discrepancy, note it
        if abs(sum_of_components - total_obj_value) > 1:
            print("\nNote: The difference between sum of components and objective value "
                  "is due to the optimization model's handling of non-negative penalty constraints.")

def main():
    """
    Main entry point: 
      - Reads from JSON if available, else uses sample data
      - Builds and solves the BeerCupOptimizer model
      - Outputs reports
    """
    json_file_path = r"C:\Users\ashwi\Downloads\BEER Cup\R3\New folder\updated_delay_port_destination_v2_sea.json"

    # Check if JSON file is available; else fallback to sample data
    if not os.path.isfile(json_file_path):
        print(f"JSON file not found at path: {json_file_path}")
        print("Using sample data for demonstration.")

        # Minimal sample data to allow the optimizer to run
        shipping_paths = [
            {
                "Port": "Port A",
                "Destination": "France",
                "Mode": "Sea",
                "Type of Beercup": "95",
                "Penalty Days without considering delay": "2",
                "Reward Days without considering delay": "5",
                "Penalty for Delay per day per batch": "20",
                "Urgency Bonus per Day": "10",
                "Transportation Cost": "150",
                "Packaging Cost/unit of Beercup type": "50",
                "Activation Cost of": "5000"
            },
            {
                "Port": "Port B",
                "Destination": "UK",
                "Mode": "Air",
                "Type of Beercup": "99",
                "Penalty Days without considering delay": "1",
                "Reward Days without considering delay": "4",
                "Penalty for Delay per day per batch": "15",
                "Urgency Bonus per Day": "8",
                "Transportation Cost": "200",
                "Packaging Cost/unit of Beercup type": "60",
                "Activation Cost of": "3000"
            },
            # Add more sample paths as needed...
        ]
    else:
        try:
            with open(json_file_path, 'r') as file:
                shipping_paths = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file: {e}")
            return
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return

        # Validate that shipping_paths is a list of dict
        if not isinstance(shipping_paths, list):
            print("JSON data is not a list of paths.")
            return

        for idx, path in enumerate(shipping_paths):
            if not isinstance(path, dict):
                print(f"Path at index {idx} is not a dictionary. Skipping.")
                shipping_paths[idx] = None

        shipping_paths = [path for path in shipping_paths if path is not None]

    # Example demands for demonstration
    demands = {
        ("France", 99): 500, ("France", 95): 500, ("France", 90): 1000,
        ("UK", 99): 750, ("UK", 95): 1750, ("UK", 90): 500,
        ("Denmark", 99): 500, ("Denmark", 95): 750, ("Denmark", 90): 750,
        ("Brazil", 99): 1000, ("Brazil", 95): 500, ("Brazil", 90): 1000,
        ("Spain", 99): 500, ("Spain", 95): 1500, ("Spain", 90): 1000
    }

    try:
        optimizer = BeerCupOptimizer(shipping_paths)
        model = optimizer.create_optimization_model(demands)
        optimizer.solve_and_report(model)
        print('\nOptimization Completed Successfully.')
    except Exception as e:
        print(f"Error during optimization: {e}")
        raise

if __name__ == "__main__":
    main()

