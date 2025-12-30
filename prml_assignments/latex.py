def print_matrix(matrix,cnt,num_matrices):
    str = "\\begin{pmatrix}\n"
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    for i in range(rows):
        str += f"{matrix[i][0]}"
        for j in range(1,cols):
            str += f"&{matrix[i][j]}"
        str += "\\\\ \n"
    if cnt < num_matrices-1:
        str += "\\end{pmatrix} \n"
    else:
        str += "\\end{pmatrix}\\\\ \n"
    return str

def print_accuracies(train_accuracy,test_accuracy):
   str = f"""\\begin{{table}}[h!]
    \\centering
    \\small
    \\caption{{Classification accuracies on train and test data}}
    \\begin{{tabular}}{{|c|c|}}
    \\hline
    \\textbf{{Train Accuracy}} & \\textbf{{Test Accuracy}}\\\\
    \\hline
    {train_accuracy} & {test_accuracy}\\\\
    \\hline
    \\end{{tabular}}
\\end{{table}}"""
   return str
   
def print_confusion_matrix(confusion_matrix_train,confusion_matrix_test):
    cnt = 0
    str = f"""\\begin{{table}}[h!]
    \\centering
    \\small
    \\caption{{Confusion matrix for train and test data}}
    \\begin{{tabular}}{{|c|c|}}
    \\hline
    \\textbf{{Train Data}} & \\textbf{{Test Data}}\\\\
    \\hline\n"""
    str += print_matrix(confusion_matrix_train,cnt,2)
    cnt += 1
    str += "&\n"
    str += print_matrix(confusion_matrix_test,cnt,2)
    str += """    \\hline
    \\end{tabular}
\\end{table}"""
    return str

def print_accuracies(train_accuracy,val_accuracy,test_accuracy):
    str = f"""\\begin{{table}}[h!]
    \\centering
    \\small
    \\caption{{Classification accuracies on train, validation and test data}}
    \\begin{{tabular}}{{|c|c|c|}}
    \\hline
    \\textbf{{Train Accuracy}} & \\textbf{{Validation Accuracy}} & \\textbf{{Test Accuracy}}\\\\
    \\hline
    {train_accuracy} & {val_accuracy} & {test_accuracy}\\\\
    \\hline
    \\end{{tabular}}
\\end{{table}}"""
    return str

def bounded_and_unbounded(bounded,unbounded):
    str = f"""\\begin{{table}}[h!]
    \\centering
    \\small
    \\caption{{Percentage of bounded and unbounded support vectors}}
    \\begin{{tabular}}{{|c|c|}}
    \\hline
    \\textbf{{Bounded}} & \\textbf{{Unbounded}}\\\\
    \\hline
    {bounded} & {unbounded}\\\\
    \\hline
    \\end{{tabular}}
\\end{{table}}"""
    return str