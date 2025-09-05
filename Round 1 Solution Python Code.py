from docplex.mp.model import Model
import pandas as pd

# -----------------------------
# Set Pandas Display Options
# -----------------------------
pd.set_option('display.float_format', '{:,.2f}'.format)

# -----------------------------
# Model Initialization
# -----------------------------
model = Model(name='BeerCupOptimization')

# -----------------------------
# Data Definitions
# -----------------------------

suppliers = {
    'Montana': {'capacity': 1000, 'batch_size': 50},
    'Michigan': {'capacity': 2000, 'batch_size': 100},
    'Colorado': {'capacity': 1500, 'batch_size': 75},
    'Illinois': {'capacity': 750, 'batch_size': 50},
    'California': {'capacity': 1500, 'batch_size': 30}
}

factories = {
    'Washington': {'capacity': 200, 'fixed_cost': 750, 'variable_cost': 20},
    'Iowa': {'capacity': 150, 'fixed_cost': 800, 'variable_cost': 15},
    'Kansas': {'capacity': 175, 'fixed_cost': 1000, 'variable_cost': 12},
    'Wisconsin': {'capacity': 180, 'fixed_cost': 650, 'variable_cost': 22},
    'Kentucky': {'capacity': 130, 'fixed_cost': 700, 'variable_cost': 21}
}

warehouses = {
    'NorthDakota': {'capacity': 150},
    'Nevada': {'capacity': 100},
    'Tennessee': {'capacity': 300},
    'Arizona': {'capacity': 150},
    'NewMexico': {'capacity': 250}
}

supplier_factory_paths = {
    ('Montana', 'Washington'): {'distance': 190, 'cost_per_km': 50, 'batch_size': 50},
    ('Montana', 'Wisconsin'): {'distance': 50, 'cost_per_km': 90, 'batch_size': 50},
    ('Montana', 'Iowa'): {'distance': 105, 'cost_per_km': 75, 'batch_size': 50},
    ('Montana', 'Kansas'): {'distance': 140, 'cost_per_km': 75, 'batch_size': 50},
    ('Montana', 'Kentucky'): {'distance': 220, 'cost_per_km': 45, 'batch_size': 50},
    
    ('Michigan', 'Washington'): {'distance': 210, 'cost_per_km': 60, 'batch_size': 100},
    ('Michigan', 'Wisconsin'): {'distance': 10, 'cost_per_km': 75, 'batch_size': 100},
    ('Michigan', 'Iowa'): {'distance': 45, 'cost_per_km': 35, 'batch_size': 100},
    ('Michigan', 'Kansas'): {'distance': 160, 'cost_per_km': 40, 'batch_size': 100},
    ('Michigan', 'Kentucky'): {'distance': 240, 'cost_per_km': 45, 'batch_size': 100},
    
    ('Colorado', 'Washington'): {'distance': 80, 'cost_per_km': 25, 'batch_size': 75},
    ('Colorado', 'Wisconsin'): {'distance': 160, 'cost_per_km': 30, 'batch_size': 75},
    ('Colorado', 'Iowa'): {'distance': 155, 'cost_per_km': 45, 'batch_size': 75},
    ('Colorado', 'Kansas'): {'distance': 80, 'cost_per_km': 50, 'batch_size': 75},
    ('Colorado', 'Kentucky'): {'distance': 80, 'cost_per_km': 80, 'batch_size': 75},
    
    ('Illinois', 'Washington'): {'distance': 145, 'cost_per_km': 80, 'batch_size': 50},
    ('Illinois', 'Wisconsin'): {'distance': 65, 'cost_per_km': 60, 'batch_size': 50},
    ('Illinois', 'Iowa'): {'distance': 70, 'cost_per_km': 30, 'batch_size': 50},
    ('Illinois', 'Kansas'): {'distance': 95, 'cost_per_km': 45, 'batch_size': 50},
    ('Illinois', 'Kentucky'): {'distance': 175, 'cost_per_km': 70, 'batch_size': 50},
    
    ('California', 'Washington'): {'distance': 85, 'cost_per_km': 65, 'batch_size': 30},
    ('California', 'Wisconsin'): {'distance': 185, 'cost_per_km': 20, 'batch_size': 30},
    ('California', 'Iowa'): {'distance': 180, 'cost_per_km': 70, 'batch_size': 30},
    ('California', 'Kansas'): {'distance': 85, 'cost_per_km': 55, 'batch_size': 30},
    ('California', 'Kentucky'): {'distance': 65, 'cost_per_km': 80, 'batch_size': 30}
}

factory_warehouse_paths = {
    ('Washington', 'NorthDakota'): {'distance': 105, 'cost_per_km': 65, 'batch_size': 30},
    ('Washington', 'Nevada'): {'distance': 175, 'cost_per_km': 30, 'batch_size': 25},
    ('Washington', 'Tennessee'): {'distance': 160, 'cost_per_km': 20, 'batch_size': 20},
    ('Washington', 'Arizona'): {'distance': 160, 'cost_per_km': 55, 'batch_size': 15},
    ('Washington', 'NewMexico'): {'distance': 215, 'cost_per_km': 80, 'batch_size': 25},
    
    ('Wisconsin', 'NorthDakota'): {'distance': 135, 'cost_per_km': 55, 'batch_size': 30},
    ('Wisconsin', 'Nevada'): {'distance': 75, 'cost_per_km': 40, 'batch_size': 25},
    ('Wisconsin', 'Tennessee'): {'distance': 80, 'cost_per_km': 70, 'batch_size': 20},
    ('Wisconsin', 'Arizona'): {'distance': 130, 'cost_per_km': 50, 'batch_size': 15},
    ('Wisconsin', 'NewMexico'): {'distance': 45, 'cost_per_km': 65, 'batch_size': 25},
    
    ('Iowa', 'NorthDakota'): {'distance': 150, 'cost_per_km': 35, 'batch_size': 30},
    ('Iowa', 'Nevada'): {'distance': 130, 'cost_per_km': 75, 'batch_size': 25},
    ('Iowa', 'Tennessee'): {'distance': 45, 'cost_per_km': 60, 'batch_size': 20},
    ('Iowa', 'Arizona'): {'distance': 185, 'cost_per_km': 25, 'batch_size': 15},
    ('Iowa', 'NewMexico'): {'distance': 100, 'cost_per_km': 50, 'batch_size': 25},
    
    ('Kansas', 'NorthDakota'): {'distance': 105, 'cost_per_km': 70, 'batch_size': 30},
    ('Kansas', 'Nevada'): {'distance': 125, 'cost_per_km': 60, 'batch_size': 25},
    ('Kansas', 'Tennessee'): {'distance': 110, 'cost_per_km': 45, 'batch_size': 20},
    ('Kansas', 'Arizona'): {'distance': 140, 'cost_per_km': 80, 'batch_size': 15},
    ('Kansas', 'NewMexico'): {'distance': 165, 'cost_per_km': 30, 'batch_size': 25},
    
    ('Kentucky', 'NorthDakota'): {'distance': 105, 'cost_per_km': 30, 'batch_size': 30},
    ('Kentucky', 'Nevada'): {'distance': 205, 'cost_per_km': 55, 'batch_size': 25},
    ('Kentucky', 'Tennessee'): {'distance': 190, 'cost_per_km': 65, 'batch_size': 20},
    ('Kentucky', 'Arizona'): {'distance': 190, 'cost_per_km': 20, 'batch_size': 15},
    ('Kentucky', 'NewMexico'): {'distance': 245, 'cost_per_km': 80, 'batch_size': 25}
}

units_per_beer_cup = {
    'MK-V': {'99': 10, '95': 8, '90': 7},
    'TAINS': {'99': 6, '95': 6, '90': 5},
    'Inferno': {'99': 8, '95': 8, '90': 8},
    'SkyTracker-X': {'99': 5, '95': 4, '90': 2},
    'Hellstrike': {'99': 7, '95': 5, '90': 4}
}

unit_cost_by_factory = {
    'MK-V': {'Montana': 50, 'California': 45, 'Michigan': 35, 'Illinois': 40, 'Colorado': 50},
    'TAINS': {'Montana': 150, 'California': 120, 'Michigan': 135, 'Illinois': 140, 'Colorado': 130},
    'Inferno': {'Montana': 65, 'California': 45, 'Michigan': 50, 'Illinois': 55, 'Colorado': 60},
    'SkyTracker-X': {'Montana': 15, 'California': 20, 'Michigan': 30, 'Illinois': 10, 'Colorado': 15},
    'Hellstrike': {'Montana': 10, 'California': 20, 'Michigan': 15, 'Illinois': 10, 'Colorado': 20}
}

# -----------------------------
# Decision Variables
# -----------------------------
trips_supplier_factory = model.integer_var_dict(
    supplier_factory_paths.keys(), name='trips_supplier_factory'
)

trips_factory_warehouse = model.integer_var_dict(
    factory_warehouse_paths.keys(), name='trips_factory_warehouse'
)

production = model.integer_var_dict(
    factories.keys(), name='production'
)

production_types = model.integer_var_dict(
    ((f, t) for f in factories for t in ['99', '95', '90']),
    name='production_types'
)

factory_operational = model.binary_var_dict(
    factories.keys(), name='factory_operational'
)

raw_material_purchase = model.continuous_var_dict(
    ((f, s, component) for f in factories for s in suppliers for component in units_per_beer_cup),
    name='raw_material_purchase'
)

total_cost = model.continuous_var(name='total_cost')

# -----------------------------
# Constraints
# -----------------------------
big_M_supplier_factory = {
    (s, f): suppliers[s]['capacity'] // supplier_factory_paths[(s, f)]['batch_size']
    for (s, f) in supplier_factory_paths
}

big_M_factory_warehouse = {
    (f, w): factories[f]['capacity'] // factory_warehouse_paths[(f, w)]['batch_size']
    for (f, w) in factory_warehouse_paths
}

# Supplier-to-Factory Trips Constraints
for supplier, data in suppliers.items():
    for factory in factories:
        if (supplier, factory) in supplier_factory_paths:
            model.add_constraint(
                trips_supplier_factory[(supplier, factory)] <= big_M_supplier_factory[(supplier, factory)] * factory_operational[factory],
                ctname=f'trips_supplier_factory_limit_{supplier}_{factory}'
            )
        else:
            model.add_constraint(
                trips_supplier_factory[(supplier, factory)] == 0,
                ctname=f'no_path_{supplier}_{factory}'
            )

# Factory Production Constraints
for factory, data in factories.items():
    model.add_constraint(
        production[factory] <= data['capacity'],
        ctname=f'production_capacity_{factory}'
    )
    model.add_constraint(
        production_types[(factory, '99')] + production_types[(factory, '95')] + production_types[(factory, '90')] == production[factory],
        ctname=f'production_sum_{factory}'
    )
    model.add_constraint(
        production[factory] <= data['capacity'] * factory_operational[factory],
        ctname=f'operational_link_{factory}'
    )

# Raw Material Requirements
for factory in factories:
    for component in units_per_beer_cup:
        required = model.sum(
            units_per_beer_cup[component][t] * production_types[(factory, t)] for t in ['99', '95', '90']
        )
        total_purchased = model.sum(
            raw_material_purchase[(factory, s, component)] for s in suppliers
        )
        M = 1000000
        model.add_constraint(
            total_purchased >= required - M * (1 - factory_operational[factory]),
            ctname=f'raw_requirement_lower_{factory}_{component}'
        )
        model.add_constraint(
            total_purchased <= required + M * (1 - factory_operational[factory]),
            ctname=f'raw_requirement_upper_{factory}_{component}'
        )

# Factory to Warehouse Trips Constraints
for (factory, warehouse), data in factory_warehouse_paths.items():
    model.add_constraint(
        trips_factory_warehouse[(factory, warehouse)] <= big_M_factory_warehouse[(factory, warehouse)] * factory_operational[factory],
        ctname=f'trips_factory_warehouse_limit_{factory}_{warehouse}'
    )

# All production must be shipped to warehouses
for factory in factories:
    model.add_constraint(
        production[factory] == model.sum(
            factory_warehouse_paths[(factory, w)]['batch_size'] * trips_factory_warehouse[(factory, w)]
            for w in warehouses if (factory, w) in factory_warehouse_paths
        ),
        ctname=f'transport_production_eq_{factory}'
    )

# Supplier Capacity
for supplier, data in suppliers.items():
    total_supplied = model.sum(
        supplier_factory_paths[(supplier, f)]['batch_size'] * trips_supplier_factory[(supplier, f)]
        for f in factories if (supplier, f) in supplier_factory_paths
    )
    model.add_constraint(
        total_supplied <= data['capacity'],
        ctname=f'supplier_capacity_{supplier}'
    )

# Warehouse Capacity
for warehouse, data in warehouses.items():
    total_received = model.sum(
        factory_warehouse_paths[(f, warehouse)]['batch_size'] * trips_factory_warehouse[(f, warehouse)]
        for f in factories if (f, warehouse) in factory_warehouse_paths
    )
    model.add_constraint(
        total_received <= data['capacity'],
        ctname=f'warehouse_capacity_{warehouse}'
    )

# Supplier Raw Material Capacity
for s in suppliers:
    for component in units_per_beer_cup:
        total_bought = model.sum(raw_material_purchase[(f, s, component)] for f in factories)
        model.add_constraint(
            total_bought <= suppliers[s]['capacity'],
            ctname=f'supplier_raw_capacity_{s}_{component}'
        )

# Link raw_material_purchase to trips_supplier_factory
for (f, s, component) in raw_material_purchase:
    model.add_constraint(
        raw_material_purchase[(f, s, component)] <= supplier_factory_paths[(s, f)]['batch_size'] * trips_supplier_factory[(s, f)],
        ctname=f'raw_purchase_transport_link_{f}_{s}_{component}'
    )

# -----------------------------
# Updated Constraints
# -----------------------------

# Total Production Constraint: Exactly 835
model.add_constraint(
    model.sum(production[f] for f in factories) == 835,
    ctname='production_target_exact'
)

# Percentage Constraints for '99' and '95' Types

# Calculate total production
total_production = model.sum(production[f] for f in factories)

# Calculate total '99' and '95' production
total_99 = model.sum(production_types[(f, '99')] for f in factories)
total_95 = model.sum(production_types[(f, '95')] for f in factories)

# Constraint: '99' type between 15% to 20%
model.add_constraint(
    total_99 >= 0.17 * total_production,
    ctname='min_15_percent_99'
)
model.add_constraint(
    total_99 <= 0.20 * total_production,
    ctname='max_20_percent_99'
)

# Constraint: '95' type between 40% to 65%
model.add_constraint(
    total_95 >= 0.40 * total_production,
    ctname='min_40_percent_95'
)
model.add_constraint(
    total_95 <= 0.65 * total_production,
    ctname='max_65_percent_95'
)

# Constraint: Sum of '99' and '95' types is at least 65% of total production
model.add_constraint(
    total_99 + total_95 >= 0.74 * total_production,
    ctname='sum_99_95_at_least_65_percent'
)

# -----------------------------
# Cost Calculations
# -----------------------------
factory_production_cost = model.sum(
    factories[f]['fixed_cost'] * factory_operational[f] + factories[f]['variable_cost'] * production[f]
    for f in factories
)

raw_material_cost = model.sum(
    raw_material_purchase[(f, s, component)] * unit_cost_by_factory[component][s]
    for (f, s, component) in raw_material_purchase
)

supplier_to_factory_transport_cost = model.sum(
    supplier_factory_paths[(s, f)]['distance'] * supplier_factory_paths[(s, f)]['cost_per_km'] * trips_supplier_factory[(s, f)]
    for (s, f) in supplier_factory_paths
)

factory_to_warehouse_transport_cost = model.sum(
    factory_warehouse_paths[(f, w)]['distance'] * factory_warehouse_paths[(f, w)]['cost_per_km'] * trips_factory_warehouse[(f, w)]
    for (f, w) in factory_warehouse_paths
)

model.add_constraint(
    total_cost == factory_production_cost + raw_material_cost +
    supplier_to_factory_transport_cost + factory_to_warehouse_transport_cost,
    ctname='total_cost_constraint'
)

# -----------------------------
# Objective Function
# -----------------------------
model.minimize(total_cost)

# -----------------------------
# Solve the Model
# -----------------------------
solution = model.solve(log_output=True)

if solution:
    # -----------------------------
    # Tabulated Outputs
    # -----------------------------
    # 1. Total Production per Factory
    factory_production_df = pd.DataFrame([
        {'Factory': factory, 'Total Production': production[factory].solution_value}
        for factory in factories
    ])
    factory_production_df['Total Production'] = factory_production_df['Total Production'].map('{:,.0f}'.format)
    print("\nTotal Number of Beer Cups Produced at Each Factory:\n", factory_production_df)
    
    # 2. Total Production by Type with Percentage
    total_production_99 = sum(production_types[(f, '99')].solution_value for f in factories)
    total_production_95 = sum(production_types[(f, '95')].solution_value for f in factories)
    total_production_90 = sum(production_types[(f, '90')].solution_value for f in factories)
    total_production_val = total_production_99 + total_production_95 + total_production_90
    
    # Compute Percentages
    percent_99 = (total_production_99 / total_production_val) * 100 if total_production_val > 0 else 0
    percent_95 = (total_production_95 / total_production_val) * 100 if total_production_val > 0 else 0
    percent_90 = (total_production_90 / total_production_val) * 100 if total_production_val > 0 else 0
    
    # Create DataFrame with Percentages
    total_production_by_type = [
        {'Type': '99', 'Total Produced': total_production_99, 'Percentage': percent_99},
        {'Type': '95', 'Total Produced': total_production_95, 'Percentage': percent_95},
        {'Type': '90', 'Total Produced': total_production_90, 'Percentage': percent_90}
    ]
    total_production_df = pd.DataFrame(total_production_by_type)
    total_production_df['Total Produced'] = total_production_df['Total Produced'].map('{:,.0f}'.format)
    total_production_df['Percentage'] = total_production_df['Percentage'].map('{:,.2f}%'.format)
    print("\nTotal Number of 90, 95, 99 Beercups Produced Across All Factories:\n", total_production_df)
    
    # 3. Total Quantity of Component Consumption
    component_consumption = []
    for component in units_per_beer_cup:
        for t in ['99', '95', '90']:
            total_used = sum(units_per_beer_cup[component][t] * production_types[(f, t)].solution_value for f in factories)
            component_consumption.append({'Component': component, 'Type': t, 'Total Consumed': total_used})
    component_consumption_df = pd.DataFrame(component_consumption)
    component_consumption_df['Total Consumed'] = component_consumption_df['Total Consumed'].map('{:,.0f}'.format)
    print("\nTotal Quantity of Component Consumption:\n", component_consumption_df)
    
    # 4. Quantity of Each Component Bought from Each Supplier
    component_bought = []
    for component in units_per_beer_cup:
        for supplier in suppliers:
            total_bought = sum(raw_material_purchase[(f, supplier, component)].solution_value for f in factories)
            component_bought.append({'Component': component, 'Supplier': supplier, 'Total Bought': total_bought})
    component_bought_df = pd.DataFrame(component_bought)
    component_bought_df['Total Bought'] = component_bought_df['Total Bought'].map('{:,.0f}'.format)
    print("\nQuantity of Each Component Bought from Each Supplier:\n", component_bought_df)
    
    # 5. Number of Trips Made Along All Paths
    trips_data = []
    for (s, f) in supplier_factory_paths:
        trips = trips_supplier_factory[(s, f)].solution_value
        trips_data.append({'Path': f'{s} -> {f}', 'Number of Trips': trips})
    for (f, w) in factory_warehouse_paths:
        trips = trips_factory_warehouse[(f, w)].solution_value
        trips_data.append({'Path': f'{f} -> {w}', 'Number of Trips': trips})
    trips_df = pd.DataFrame(trips_data)
    trips_df['Number of Trips'] = trips_df['Number of Trips'].map('{:,.0f}'.format)
    print("\nNumber of Trips Made Along All Paths:\n", trips_df)
    
    # 6. Total Cost Breakdown
    factory_production_cost_val = sum(
        factories[f]['fixed_cost'] * factory_operational[f].solution_value +
        factories[f]['variable_cost'] * production[f].solution_value
        for f in factories
    )

    raw_material_cost_val = sum(
        raw_material_purchase[(f, s, component)].solution_value * unit_cost_by_factory[component][s]
        for (f, s, component) in raw_material_purchase
    )

    supplier_to_factory_transport_cost_val = sum(
        supplier_factory_paths[(s, f)]['distance'] * supplier_factory_paths[(s, f)]['cost_per_km'] * trips_supplier_factory[(s, f)].solution_value
        for (s, f) in supplier_factory_paths
    )

    factory_to_warehouse_transport_cost_val = sum(
        factory_warehouse_paths[(f, w)]['distance'] * factory_warehouse_paths[(f, w)]['cost_per_km'] * trips_factory_warehouse[(f, w)].solution_value
        for (f, w) in factory_warehouse_paths
    )

    total_cost_val = total_cost.solution_value

    total_cost_breakdown = {
        'Factory Production Cost': factory_production_cost_val,
        'Raw Material Cost': raw_material_cost_val,
        'Supplier to Factory Transportation Cost': supplier_to_factory_transport_cost_val,
        'Factory to Warehouse Transportation Cost': factory_to_warehouse_transport_cost_val,
        'Total Cost': total_cost_val
    }

    total_cost_df = pd.DataFrame(list(total_cost_breakdown.items()), columns=['Cost Component', 'Amount'])
    total_cost_df['Amount'] = total_cost_df['Amount'].map('${:,.2f}'.format)
    print("\nTotal Cost Breakdown:\n", total_cost_df)

    # 7. Factory Operational Status
    factory_status_df = pd.DataFrame([
        {'Factory': factory, 'Operational': 'Yes' if factory_operational[factory].solution_value > 0.5 else 'No'}
        for factory in factories
    ])
    print("\nFactory Operational Status:\n", factory_status_df)
    
    # 8. Cost of Raw Material Utilized
    print(f"\nCost of Raw Material Utilized: ${raw_material_cost_val:,.2f}")
    
    # -----------------------------
    # Additional Cost Allocations per Type
    # -----------------------------
    
    # 1. Total cost per unit (all types combined)
    total_cost_per_unit_all = total_cost_val / total_production_val
    print(f"\nTotal Cost per Unit (All Beer Cups): ${total_cost_per_unit_all:,.2f}")
    
    # Allocate costs per type:
    # Step A: Compute total component usage by each type
    component_usage_by_type = {t: {} for t in ['99', '95', '90']}
    # Total units of each component (across all types)
    total_component_usage = {c: 0 for c in units_per_beer_cup}
    for c in units_per_beer_cup:
        for t in ['99', '95', '90']:
            usage = sum(units_per_beer_cup[c][t] * production_types[(f, t)].solution_value for f in factories)
            component_usage_by_type[t][c] = usage
            total_component_usage[c] += usage

    # Step B: Compute total cost per component (raw materials)
    # raw_material_cost_val is total, but we need per component breakdown:
    # We can sum cost paid for each component:
    component_cost = {c: 0 for c in units_per_beer_cup}
    for (f, s, c) in raw_material_purchase:
        amt = raw_material_purchase[(f, s, c)].solution_value
        cost_per_unit = unit_cost_by_factory[c][s]
        component_cost[c] += amt * cost_per_unit

    # Step C: Allocate raw material cost to each type by proportion of component usage
    raw_material_cost_by_type = {t: 0 for t in ['99', '95', '90']}
    for c in component_cost:
        if total_component_usage[c] > 0:
            for t in ['99', '95', '90']:
                fraction = component_usage_by_type[t][c] / total_component_usage[c] if total_component_usage[c] > 0 else 0
                raw_material_cost_by_type[t] += component_cost[c] * fraction

    # Step D: Allocate supplier-to-factory transport cost by raw material usage fraction
    # Total raw material usage (sum of all components)
    total_rm_usage = sum(total_component_usage[c] for c in total_component_usage)
    # If total_rm_usage = 0 (edge case), avoid division by zero
    supplier_to_factory_cost_by_type = {t: 0 for t in ['99', '95', '90']}
    if total_rm_usage > 0:
        for t in ['99', '95', '90']:
            # Share of raw materials for this type
            type_rm_share = sum(component_usage_by_type[t][c] for c in component_usage_by_type[t]) / total_rm_usage
            supplier_to_factory_cost_by_type[t] = supplier_to_factory_transport_cost_val * type_rm_share

    # Step E: Allocate factory-to-warehouse transport cost by final product count
    factory_to_warehouse_cost_by_type = {t: 0 for t in ['99', '95', '90']}
    if total_production_val > 0:
        factory_to_warehouse_cost_by_type['99'] = factory_to_warehouse_transport_cost_val * (total_production_99 / total_production_val)
        factory_to_warehouse_cost_by_type['95'] = factory_to_warehouse_transport_cost_val * (total_production_95 / total_production_val)
        factory_to_warehouse_cost_by_type['90'] = factory_to_warehouse_transport_cost_val * (total_production_90 / total_production_val)

    # Step F: Allocate factory production cost based on each factory's mix
    factory_cost_by_type = {t: 0 for t in ['99', '95', '90']}
    for f in factories:
        # Factory cost
        f_cost = factories[f]['fixed_cost'] * factory_operational[f].solution_value + factories[f]['variable_cost'] * production[f].solution_value
        f_total = production[f].solution_value
        if f_total > 0:
            f_99 = production_types[(f, '99')].solution_value
            f_95 = production_types[(f, '95')].solution_value
            f_90 = production_types[(f, '90')].solution_value
            factory_cost_by_type['99'] += f_cost * (f_99 / f_total)
            factory_cost_by_type['95'] += f_cost * (f_95 / f_total)
            factory_cost_by_type['90'] += f_cost * (f_90 / f_total)

    # Now sum all costs for each type:
    total_cost_by_type = {t: 0 for t in ['99', '95', '90']}
    for t in ['99', '95', '90']:
        total_cost_by_type[t] = (raw_material_cost_by_type[t] +
                                 supplier_to_factory_cost_by_type[t] +
                                 factory_to_warehouse_cost_by_type[t] +
                                 factory_cost_by_type[t])

    # Compute per-unit cost for each type
    per_unit_99 = total_cost_by_type['99'] / total_production_99 if total_production_99 > 0 else 0
    per_unit_95 = total_cost_by_type['95'] / total_production_95 if total_production_95 > 0 else 0
    per_unit_90 = total_cost_by_type['90'] / total_production_90 if total_production_90 > 0 else 0

    print(f"\nPer-Unit Cost for '99' Type: ${per_unit_99:,.2f}")
    print(f"Per-Unit Cost for '95' Type: ${per_unit_95:,.2f}")
    print(f"Per-Unit Cost for '90' Type: ${per_unit_90:,.2f}")
    
    # -----------------------------
    # Adding Percentages to Output Tables
    # -----------------------------
    
    # The percentage has already been added to 'total_production_df'
    # Ensure that factory_production_df includes percentages if desired
    # For this example, we'll focus on the total production percentages

    # If you wish to add percentages to factory_production_df as well, uncomment the following:

    # factory_total = sum(production[factory].solution_value for factory in factories)
    # factory_production_df['Percentage'] = factory_production_df['Total Production'].astype(float) / factory_total * 100
    # factory_production_df['Percentage'] = factory_production_df['Percentage'].map('{:,.2f}%'.format)
    # print("\nTotal Number of Beer Cups Produced at Each Factory with Percentages:\n", factory_production_df)

else:
    print("No solution found.")
