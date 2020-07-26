# Airplane-Accident-Severity-Prediction
This contains a machine learning model that will predict the severity of an airplane accident using logistic regression

# Data columns (total 12 columns):

<table>
  <tr><td>Feature</td>               <td>Total Entries</td><td>If Null Any</td><td>Object type</td></tr>
  <tr><td>Severity</td>              <td>10000</td> <td>non-null</td> <td>object</td></tr>
  <tr><td>Safety_Score               <td>10000</td> <td>non-null</td> <td>float64</td></tr>
  <tr><td>Days_Since_Inspection      <td>10000</td> <td>non-null</td> <td>int64</td></tr>
  <tr><td>Total_Safety_Complaints    <td>10000</td> <td>non-null</td> <td>int64</td></tr>
  <tr><td>Control_Metric             <td>10000</td> <td>non-null</td> <td>float64</td></tr>
  <tr><td>Turbulence_In_gforces      <td>10000</td> <td>non-null</td> <td>float64</td></tr>
  <tr><td>Cabin_Temperature          <td>10000</td> <td>non-null</td> <td>float64</td></tr>
  <tr><td>Accident_Type_Code         <td>10000</td> <td>non-null</td> <td>int64</td></tr>
  <tr><td>Max_Elevation              <td>10000</td> <td>non-null</td> <td>float64</td></tr>
  <tr> <td>Violations                <td>10000</td> <td>non-null</td> <td>int64</td></tr>
  <tr><td>Adverse_Weather_Metric     <td>10000</td> <td>non-null</td> <td>float64</td></tr>
  <tr><td>Accident_ID                <td>10000</td> <td>non-null</td> <td>int64</td></tr>
</table>


# Feature Info:
<table>
<thead>
<tr>
<th></th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<td>Columns</td>
<td>Description</td>
</tr>
<tr>
<td>Accident_ID</td>
<td>unique id assigned to each row</td>
</tr>
<tr>
<td>Accident_Type_Code</td>
<td>the type of accident (factor, not numeric)</td>
</tr>
<tr>
<td>Cabin_Temperature</td>
<td>the last recorded temperature before the incident, measured in degrees fahrenheit</td>
</tr>
<tr>
<td>Turbulence_In_gforces</td>
<td>the recorded/estimated turbulence experienced during the accident</td>
</tr>
<tr>
<td>Control_Metric</td>
<td>an estimation of how much control the pilot had during the incident given the factors at play</td>
</tr>
<tr>
<td>Total_Safety_Complaints</td>
<td>number of complaints from mechanics prior to the accident</td>
</tr>
<tr>
<td>Days_Since_Inspection</td>
<td>how long the plane went without inspection before the incident</td>
</tr>
<tr>
<td>Safety_Score</td>
<td>a measure of how safe the plane was deemed to be</td>
</tr>
<tr>
<td>Violations</td>
<td>number of violations that the aircraft received during inspections</td>
</tr>
<tr>
<td>Severity</td>
<td>a description (4 level factor) on the severity of the crash [Target]</td>
</tr>
</tbody>
</table>
