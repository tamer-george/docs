<h1>Data Visualization CheatSheet (everything you might need )</h1>
<p>
The project include module with functions to summarize and analysis dataset in order 
to understand properties and relationships.


</p>
<h2> DESCRIPTIVE  MODULE</h2>
<h3>Answer the question (What Happen?)</h3>

Includes one module:

            
  - <i>pyeda.py</i><br>
Includes simple data visualization functions that might be useful in general cases.<br><br>
  

          Example:
          
          import descriptive.pyeda as eda 

          data = eda.import_dataset("data.csv") 
          eda.visualize_distribution_of_numeric_col(data, "column_name")
          
    
           