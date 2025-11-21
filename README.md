# Beer Cup Supply Chain Optimization

## Overview

This project tackles a complex supply chain optimization problem for beer cup distribution across multiple international destinations. The challenge involves optimizing the entire supply chain from raw material suppliers through factories and warehouses to final destinations, while managing various constraints like capacity limits, transportation costs, and product quality requirements.

## Background

The beer cup industry faces unique challenges in supply chain management due to varying quality requirements, multiple transportation modes, port capacity limitations, and time-sensitive delivery schedules. This project was developed as part of a multi-round optimization competition where each round introduced additional complexity and constraints.

## Problem Statement

The core challenge is to minimize total supply chain costs while ensuring:
- All customer demands are met across different destinations (France, UK, Denmark, Brazil, Spain)
- Three different beer cup quality types (99%, 95%, 90%) are produced in optimal ratios
- Port capacity constraints are respected
- Transportation delays and penalties are managed effectively
- Raw material sourcing is optimized across multiple suppliers

## Solution Approach

This repository contains solutions for multiple rounds of increasing complexity:

### Round 1: Basic Supply Chain Optimization

**Key Features:**
- Multi-stage supply chain from suppliers → factories → warehouses
- Production planning across 5 factories (Washington, Iowa, Kansas, Wisconsin, Kentucky)
- Raw material sourcing from 5 suppliers (Montana, Michigan, Colorado, Illinois, California)
- Distribution to 5 warehouses (North Dakota, Nevada, Tennessee, Arizona, New Mexico)
- Component-based production system for beer cups

**Optimization Goals:**
- Minimize total costs (production + raw materials + transportation)
- Meet production target of exactly 835 units
- Balance production across different quality types:
  - Type 99: 17-20% of total production
  - Type 95: 40-65% of total production
  - Combined 99 + 95: at least 74% of total production

**Technologies Used:**
- IBM CPLEX (via docplex) for mixed-integer programming
- Python for modeling and data processing
- Pandas for data analysis and presentation

### Round 3: Advanced Multi-Port Distribution

**Key Features:**
- Complex port-to-destination routing across 10 ports (A through J)
- Multiple transportation modes (sea and air)
- Dynamic penalty and reward systems based on delivery timing
- Port-specific constraints and special conditions:
  - Port C: 2-day delay
  - Port J: 3-day delay
  - Port F: Quadratic penalty calculation
  - Port H: Reduced capacity (0.7x standard)
  - Port A, B, E, G, I: Special miscellaneous costs
- Minimum 5 ports must be activated
- Route-based activation costs

**Advanced Optimization Features:**
- Batch-based shipment planning (scale factor: 50 units per batch)
- Urgency bonus for early deliveries
- Penalty costs for delayed shipments
- Port capacity management
- Mode-specific activation requirements

**Technologies Used:**
- Gurobi Optimizer for advanced mixed-integer programming
- Python with custom optimizer class architecture
- PrettyTable for formatted output presentation
- JSON-based path configuration system

## Results

### Round 1 Achievements

The optimization successfully:
- **Total Production**: 835 beer cups produced across all factories
- **Cost Efficiency**: Achieved optimal balance between fixed and variable costs
- **Quality Distribution**: Met all percentage constraints for different beer cup types
- **Factory Utilization**: Identified optimal factory activation strategy
- **Transportation Efficiency**: Minimized trips while meeting all capacity constraints

**Cost Breakdown:**
- Factory production costs (fixed + variable)
- Raw material procurement costs
- Supplier-to-factory transportation
- Factory-to-warehouse distribution

### Round 3 Achievements

The solution optimizes complex international distribution:
- **Total Batches Shipped**: Optimized across all active ports
- **Active Ports**: 5+ ports strategically selected from 10 available options
- **Delivery Performance**: Maximized urgency bonuses while minimizing delay penalties
- **Cost Components**:
  - Transportation costs (sea and air modes)
  - Packaging costs per beer cup type
  - Activation costs per route
  - Penalty costs for delays
  - Miscellaneous port-specific charges
  - Reward credits for early/on-time deliveries

**Destination Coverage:**
- France: All three quality types delivered
- UK: Complete coverage across quality spectrum
- Denmark: Full demand satisfaction
- Brazil: All requirements met
- Spain: Complete order fulfillment

## Project Structure

```
├── README.md                           # Project documentation
├── Round 1 Solution Python Code.py    # Basic supply chain optimization
├── R1 Solution Excel.xlsm              # Round 1 supporting data and validation
├── R3 Solution Python Code.py          # Advanced multi-port distribution
├── Round 3 Solution Excel.xlsx         # Round 3 data and results
├── Round 3 Backstory-output.pdf        # Problem background and context
├── Round 3 Rulebook-output.pdf         # Detailed constraints and rules
├── Rulebook Round 2 -output.pdf        # Round 2 specifications
└── Rulebook Round 2 .pdf               # Round 2 documentation
```

## Technical Implementation

### Round 1 Implementation Details

The solution uses IBM CPLEX through the docplex library to model:
- **Decision Variables**:
  - Integer variables for trips between suppliers-factories and factories-warehouses
  - Integer variables for production quantities
  - Binary variables for factory operational status
  - Continuous variables for raw material purchases
  
- **Constraints**:
  - Capacity constraints at suppliers, factories, and warehouses
  - Production mix requirements
  - Raw material balance equations
  - Batch size limitations
  - Operational linking constraints

### Round 3 Implementation Details

The advanced solution implements a sophisticated optimization model:
- **BeerCupOptimizer Class**: Custom optimizer with modular constraint handling
- **Path Management**: Dynamic routing across 10 ports and 5 destinations
- **Cost Calculation**: Multi-component cost function with penalties and rewards
- **Constraint Categories**:
  - Activation constraints (minimum ports, route activation)
  - Capacity constraints (port-specific limits)
  - Demand constraints (destination-specific requirements)
  - Special conditions (port-specific delays and multipliers)

**Key Algorithms**:
- Binary variables for route activation
- Big-M formulation for conditional constraints
- Piecewise linear approximations for complex penalty functions
- Quadratic penalty handling for Port F

## How to Run

### Prerequisites

**For Round 1:**
```bash
pip install docplex pandas
```

**For Round 3:**
```bash
pip install gurobipy prettytable
```

Note: Gurobi requires a license (free academic licenses available).

### Execution

**Round 1:**
```bash
python "Round 1 Solution Python Code.py"
```

**Round 3:**
```bash
python "R3 Solution Python Code.py"
```

**Note about data files:** The Round 3 code references a JSON file for path configuration. The code currently contains a hardcoded path that you'll need to update:
```python
# Update this path to your JSON file location
json_file_path = r"path/to/your/updated_delay_port_destination_v2_sea.json"
```

You'll need to either:
1. Update this path to point to your JSON file location
2. Modify the code to use relative paths
3. Use the sample data that's built into the code if no JSON file is found

The JSON file should contain an array of shipping path objects with the following structure:
```json
[
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
  }
]
```

## Key Insights

1. **Multi-Stage Optimization**: Breaking down the supply chain into distinct stages (supplier→factory→warehouse→destination) allows for more manageable optimization.

2. **Trade-offs**: The solutions demonstrate classic optimization trade-offs:
   - Fixed costs vs. operational efficiency
   - Transportation distance vs. cost per kilometer
   - Early delivery bonuses vs. capacity utilization
   - Port activation costs vs. routing flexibility

3. **Constraint Management**: Effective handling of complex constraints (percentages, capacities, delays) is crucial for real-world applicability.

4. **Scalability**: The batch-based approach in Round 3 allows handling of large-scale problems by reducing variable count.

## Future Enhancements

Potential improvements and extensions:
- Integration with real-time data sources for dynamic optimization
- Multi-period planning with inventory management
- Stochastic optimization for demand uncertainty
- Carbon footprint minimization alongside cost
- Machine learning for demand forecasting
- Interactive visualization dashboard for results

## License

This project is available for educational and research purposes.

## Contributing

Contributions, suggestions, and improvements are welcome! Feel free to open issues or submit pull requests.

## Acknowledgments

This project was developed as part of a supply chain optimization competition, demonstrating advanced mathematical programming techniques applied to real-world logistics challenges.