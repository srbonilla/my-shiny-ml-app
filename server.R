# Import R packages
library(shiny)
library(plotly)

# Define any Python packages needed for the app here:
PYTHON_DEPENDENCIES = c('pip', 'numpy', 'sklearn', 'joblib', 'matplotlib')

# Begin app server
shinyServer(function(input, output) {
  
  # App virtualenv setup
  
  virtualenv_dir = Sys.getenv('VIRTUALENV_NAME')
  python_path = Sys.getenv('PYTHON_PATH')
  
  # Create virtual env and install dependencies
  reticulate::virtualenv_create(envname = virtualenv_dir, python = python_path)
  reticulate::virtualenv_install(virtualenv_dir, packages = PYTHON_DEPENDENCIES, ignore_installed=TRUE)
  reticulate::use_virtualenv(virtualenv_dir, required = T)
  
  # App server logic 
  
  # Import python functions to R
  reticulate::source_python('python_functions.py')
  
  # Generate a plot of the data
  
  output$plot <- renderPlotly({
    
    # get training data
    data = load_train_data()
    x_train = data[[1]]
    y_train = data[[2]]
    
    # get dummy data
    curve_data = generate_prediction_curve()
    x_curve = curve_data[[1]]
    y_curve = curve_data[[2]]
    
    # get user input data
    X = process_x_string(input$x_string)
    x = X[,1]
    y = make_prediction(X)
    
    # Make plot
    plot_ly() %>%
      add_trace(x=x_train, y=y_train, type="scatter", mode="markers", name="train data", opacity = 0.5) %>%
      add_trace(x=x_curve, y=y_curve, mode="lines", name="model predictions") %>%
      add_trace(x=x, y=y, type="scatter", mode="markers", name="input data",
                color = I('black'), marker = list(size = 12, symbol="x")) %>%
      layout(xaxis=list(title="x variable"), yaxis=list(title="y prediction"))
    
    })
  
  # Test that the Python functions have been imported
  output$message <- renderText({
    
    # get user input data
    X = process_x_string(input$x_string)
    y = make_prediction(X)
    return(process_y_array(y))
  })
  
})