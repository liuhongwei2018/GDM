


library(shiny)
library(xgboost)

xgb_clf <- xgb.load("data/xgboost.model")



ui <- fluidPage(
  
  titlePanel(h2("Gestational Diabetes Mellitus Risk Calculator", 
                style = "color:#317eac", align = "center"), windowTitle = "GDM risk score"),
  column(6,
         fluidRow(
           wellPanel(fluidRow(  
             column(6,
                    h4("Demographic Predictor", style = "color:#317eac"), 
                    sliderInput("Age",label = 'Maternal age (year)', 
                                min = 19, max = 44, 
                                value = 26, step = 1),
                    radioButtons("Family_history", 
                                 label = "Family history of diabetes",
                                 choices = list("yes" = 1, "no" = 0),
                                 selected = 0),
                    radioButtons("Education", "Education",
                                 choices = list("High school and below" = 1,
                                                "Junior college" = 2,
                                                "University" = 3,
                                                "More than University" = 4),
                                 selected = 3),
                    numericInput("Income", label = "Income per month (RMB)", 
                                 value = 6000, step = 10),
                    numericInput("BMI", label = "pre-pregnancy BMI (kg/m2)",
                                 value = 18, step = 0.1),
                    numericInput("Gravidity", label = "Gravidity", 
                                 value = 1, step = 1),
                    radioButtons("parity", label = "Parity",
                                 choices = list("0" = 0, "at least 1" = 1), selected = 0)
             ),
             column(6, 
                    h4("Measurement Predictor",style = "color:#317eac"),
                    numericInput("Waist_circumference", 
                                 label = "Waist circumference (cm)",
                                 value = 70, step = 1),
                    numericInput("Hip_circumference",
                                 label = "Hip circumference (cm)",
                                 value = 85, step = 1),
                    numericInput("SBP", label = "SBP (mmHg)", value = 120, step = 1),
                    numericInput("DBP", label = "DBP (mmHg)", value = 80, step = 1),
                    numericInput("Fasting_glucose", 
                                 label = "Fasting plasma glucose (mmol/L)",
                                 value = 5.5, step = 0.01),
                    numericInput("ALT", label = "ALT (U/L)", value = 13,
                                 step = 0.1),
                    numericInput("Weight_gain", label = "Weight gain (kg)",
                                 value = 20, step = 0.1),
                    hr(),
                    actionButton("Calculate", "Calculate", width = 100))
             
           )
           )
           
         )
  ),
  column(6,
		 div(style = "font-family:'Times New Roman'; position:absolute; top:250px; align:center; font-size:15pt",
		     p("The risk of developing GDM is:", align="center"),textOutput("selected_var")),
         div(style = "font-family:'Times New Roman'; position:absolute; top:650px; font-size:9pt", p("Developed by Department of Epidemiology and Biostatistics, School of Public Health, Tianjin Medical University."))
)
)

server <- function(input, output) {
  
  #datasetInput <- eventReactive(input$Calculate, NULL,ignoreNULL = TRUE)
  
  output$selected_var <- renderPrint({ 

    if (input$Calculate == 0)
      cat("")
    
    input$Calculate
    z <- isolate({
      data <- xgb.DMatrix(data = matrix(c(input$Age,input$Waist_circumference,
                                          input$Hip_circumference, input$ALT, 
                                          input$Weight_gain, input$BMI, input$Income,
                                          as.numeric(input$Education), input$SBP, 
                                          input$DBP, as.numeric(input$parity), input$Gravidity,
                                          as.numeric(input$Family_history), 
                                          input$Fasting_glucose), nrow = 1))
      pred <- predict(xgb_clf, data)
      pred_cal <- 1/(1 + exp(pred * (-6.054486) + 5.500776))
    })
    
    cat(round(z,3)*100,"%")
  })
  
}  

shinyApp(ui, server)
