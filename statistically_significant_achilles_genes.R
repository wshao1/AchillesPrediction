# Install the necessary package if not already installed
if (!require("openxlsx")) {
  install.packages("openxlsx")
}

# Load the required library
library(openxlsx)

# Read the Excel file
file_path <- "AchillesPrediction/supp_1.xlsx"
wb <- loadWorkbook(file_path)

# Read the data from the specified sheet
data <- read.xlsx(wb, sheet = 4)

# Specify the column containing p-values
p_value_column <- "Pearson.P-Value.(uncorrected)"

# Apply Benjamini-Hochberg correction
data$adjusted_p_values <- p.adjust(data[[p_value_column]], method = "BH")

# Write the adjusted p-values back to the Excel file
write.xlsx(data, "supp_1.xlsx", sheetName = "table 4 - all_genes_performance", colNames = TRUE)
