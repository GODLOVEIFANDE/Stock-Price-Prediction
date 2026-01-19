rm(list=ls())

library(shiny)
library(plotly)
#install.packages("plotly")
library(reticulate)

# Configure Python environment (optional)
use_virtualenv("r-reticulate", required = TRUE)

# Source the updated LSTM prediction script
source_python("lstm_predict.py")

# Get available stocks from "data" folder
get_stock_list <- function() {
  files <- list.files("data", pattern = "\\.csv$", full.names = FALSE)
  tools::file_path_sans_ext(basename(files))
}

ui <- fluidPage(
  titlePanel("📈 Stock Price Prediction with LSTM Machine Learning Model"),

  sidebarLayout(
    sidebarPanel(
      selectInput("stock", "Select Stock:", choices = get_stock_list()),
      fileInput("new_stock", "Upload New Stock CSV:", accept = ".csv"),
      actionButton("predict", "Predict"),
      br(), br(),
      verbatimTextOutput("predInfo")
    ),

    mainPanel(
      plotlyOutput("candlePlot", height = "500px")
    )
  )
)

server <- function(input, output, session) {
  # Update stock list if new file uploaded
  observeEvent(input$new_stock, {
    req(input$new_stock)

    file <- input$new_stock
    stock_name <- tools::file_path_sans_ext(file$name)

    file.copy(file$datapath, paste0("data/", file$name), overwrite = TRUE)

    # Update dropdown
    updateSelectInput(session, "stock", choices = get_stock_list(), selected = stock_name)
    showNotification(paste("✅ Uploaded", file$name))
  })

  # Reactive prediction call
  prediction_result <- eventReactive(input$predict, {
    tryCatch({
      result <- predict_with_history(input$stock)
      if (!is.null(result$error)) stop(result$error)
      result
    }, error = function(e) {
      showNotification(paste(" Error:", e$message), type = "error")
      NULL
    })
  })

  # Render candlestick chart
  output$candlePlot <- renderPlotly({
    result <- prediction_result()
    req(result)
    
    plot_ly(
      x = result$Date,
      type = "candlestick",
      open = result$Open,
      high = result$High,
      low = result$Low,
      close = result$Close
    ) %>%
      layout(
        title = paste("Candlestick + Prediction -", input$stock),
        xaxis = list(title = "Date"),
        yaxis = list(title = "Price")
      )
  })
  
}

shinyApp(ui = ui, server = server)


