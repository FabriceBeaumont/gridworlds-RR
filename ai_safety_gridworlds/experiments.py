import plotly.express as px

# Sample data
x = [1, 2, 3, 4, 5]  # x-axis values
y = [10, 8, 6, 4, 2]  # y-axis values

# Create line graph using Plotly Express
fig = px.line(x=x, y=y)

# Customize the plot (optional)
fig.update_layout(
    title="Line Graph",
    xaxis_title="X-axis",
    yaxis_title="Y-axis"
)

# Show the plot
fig.show()