import zipfile
import requests
import io
import streamlit as st

@st.cache_data
def load_data():
    url = "https://github.com/user-attachments/files/27147217/cleaned_wildlife_strikes.zip"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    
    with z.open("cleaned_wildlife_strikes.parquet") as f:
        return pd.read_parquet(f)
    
df = load_data().copy()

###################################################################################################
#Configure settings for streamlit app
st.set_page_config(
    page_title="Aviation Wildlife Strikes",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded")

#Change to dark theme
alt.themes.enable("dark")

#Configure main sidebar
with st.sidebar:
  selected = option_menu(
    menu_title = "Menu",
    options = ["Home", "Heatmap", "Yearly Strikes", "Time", "Seasonality", "Aircraft", "Species", "Scatter Plot", "About"],
    icons = ["house", "map", "calendar", "clock", "tropical-storm", "airplane", "feather", "graph-up", "question"],
    menu_icon = "cast",
    default_index = 0,
  )
  
###################################################################################################
#Configure Home page.
if selected == "Home":
    st.title("Wildlife Strikes in Aviation Dashboard")
    st.markdown("""
        Welcome to the **Wildlife Strikes in Aviation Dashboard**! This app allows users to analyze and visualize aviation wildlife strike data from **1990 to present**. 
        Discover valuable insights into trends, seasonal patterns, and other metrics to help improve aviation safety.
        
        👉 **Get started** by selecting an option from the sidebar.
        
        Data obtained from the official 
        [**FAA Wildlife Strike Database**](https://wildlife.faa.gov/).
    """, unsafe_allow_html=True)
    st.markdown("## Key Metrics (1990 - 2026):")         

    #Create 4 columns for displaying the metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(label="Total Wildlife Strikes", value=f"{total_bird_strikes:,}")
    
    with col2:
        st.metric(label="Total Cost", value=f"${total_cost:,.0f}")
        
    with col3:
        st.metric(label="Total Cost (Inflation Adjusted)", value=f"${total_cost_infl_adj:,.0f}")
    
    with col4:
        st.metric(label="Total Fatalities", value=total_fatalities)
    
    with col5:
        st.metric(label="Total Injuries", value=total_injuries)
###################################################################################################
    #Create three columns for the bar charts side by side
    chart_col1, chart_col2, chart_col3 = st.columns(3)    
    
    with chart_col1:
        #Plot Top 10 States bar chart
        st.subheader('Top 10 States with Wildlife Strikes')
        state_count = strike_counts['state_count'].sort_values('Number of Strikes')

        fig, ax = plt.subplots()
        ax.barh(state_count['State'], state_count['Number of Strikes'], color='skyblue')
        ax.set_xlabel("Number of Wildlife Strikes")
        ax.set_ylabel("State")
        st.pyplot(fig)

    with chart_col2:
        #Plot Top 10 Airports bar chart
        st.subheader("Top 10 Airports with Wildlife Strikes")
        airport_count = strike_counts['airport_count'].sort_values('Number of Strikes')

        fig, ax = plt.subplots()
        ax.barh(airport_count['Airport'], airport_count['Number of Strikes'], color='lightgreen')
        ax.set_xlabel("Number of Wildlife Strikes")
        ax.set_ylabel("Airport")
        st.pyplot(fig)
    
    with chart_col3:
        #Plot Top 10 Carriers bar chart
        st.subheader("Top 10 Carriers with Wildlife Strikes")
        carrier_count = strike_counts['operator_count'].sort_values('Number of Strikes')

        fig, ax = plt.subplots()
        ax.barh(carrier_count["Carrier"], carrier_count["Number of Strikes"], color='orange')
        ax.set_xlabel("Number of Wildlife Strikes")
        ax.set_ylabel("Carrier")
        st.pyplot(fig)
            
###################################################################################################
#Configure Heatmap page.    
elif selected == "Heatmap":
    #Query to get coordinates data, with user option to filter year, operator, and time of day.
    base_query = """
        SELECT 
            AIRPORT_LATITUDE, AIRPORT_LONGITUDE, INCIDENT_YEAR, TIME_OF_DAY, OPERATOR, COUNT(*) AS BIRD_STRIKES
        FROM FAA_BIRD_STRIKES
    """

    #Sidebar filters
    #Add year range filter
    year_range = st.sidebar.slider("Select Year Range", min_value=1996, max_value=2025, value=(1996, 2025), step=1)
    start_year, end_year = year_range
    
    #Operators filter
    query_operators = "SELECT DISTINCT OPERATOR FROM FAA_BIRD_STRIKES ORDER BY OPERATOR"
    #Execute query
    cursor.execute(query_operators)
    operators = cursor.fetchall()
    #Create list of operators
    operator_list = ["All"] + [operator[0] for operator in operators]
    #Add sidebar to streamlit
    selected_operator = st.sidebar.selectbox("Select Operator", operator_list)
    
    #Time of Day filter
    query_time_of_day = "SELECT DISTINCT TIME_OF_DAY FROM FAA_BIRD_STRIKES ORDER BY TIME_OF_DAY"
    #Execute query
    cursor.execute(query_time_of_day)
    time_of_day = cursor.fetchall()
    #Create list for time of day
    time_of_day_list = ["All"] + [ToD[0] if ToD[0] is not None else "Unknown" for ToD in time_of_day]
    #Add sidebar to streamlit
    selected_time_of_day = st.sidebar.selectbox("Select Time of Day", time_of_day_list)

    #Initialize filters and parameters for dynamic query
    filters = []
    params = []
  
    #Apply year range filter
    filters.append("INCIDENT_YEAR BETWEEN %s AND %s")
    params.extend([start_year, end_year])

    #Apply operator filter if selected
    if selected_operator != "All":
        filters.append("OPERATOR = %s")
        params.append(selected_operator)

    #Apply time of day filter if selected
    if selected_time_of_day != "All":
        filters.append("TIME_OF_DAY = %s")
        params.append(selected_time_of_day)

    #Build final query
    where_clause = "WHERE " + " AND ".join(filters)
    query = f"{base_query} {where_clause} GROUP BY AIRPORT_LATITUDE, AIRPORT_LONGITUDE, INCIDENT_YEAR, TIME_OF_DAY, OPERATOR ORDER BY INCIDENT_YEAR"

    #Execute the query
    cursor.execute(query, params)
    airport_coords = cursor.fetchall()
    
    #Create coordinate df, convert coords to numeric
    coords_df = pd.DataFrame(airport_coords, columns=["Latitude", "Longitude", "Year", "TimeOfDay", "Operator", "Bird Strikes"])
    #Convert columns to numeric
    coords_df["Latitude"] = pd.to_numeric(coords_df["Latitude"], errors="coerce")
    coords_df["Longitude"] = pd.to_numeric(coords_df["Longitude"], errors="coerce")
    #Drop na values
    coords_df.dropna(subset=["Latitude", "Longitude"], inplace=True)

    #Create heat map
    global_map = folium.Map(location=[37.0902, -95.7129], zoom_start=2)
    heat_data = coords_df[["Latitude", "Longitude"]].values.tolist()
    HeatMap(heat_data).add_to(global_map)
    
    #Display map in app
    st.title(f"Heatmap of Global Wildlife Strikes ({start_year}-{end_year})")
    folium_static(global_map)
    
###################################################################################################
#Configure Yearly Strikes page. 
elif selected == "Yearly Strikes":
    st.title("Yearly Number of Wildlife Strikes")
    
    #Query to get bird strikes grouped by year, with user option to filter
    base_query = """
        SELECT 
            INCIDENT_YEAR, TIME_OF_DAY, OPERATOR, AIRCRAFT, COUNT(*) AS BIRD_STRIKES
        FROM FAA_BIRD_STRIKES
    """
    
    #Sidebar filters
    #Operators filter
    query_operators = "SELECT DISTINCT OPERATOR FROM FAA_BIRD_STRIKES ORDER BY OPERATOR"
    cursor.execute(query_operators)
    operators = cursor.fetchall()
    #Create list of operators
    operator_list = ["All"] + [ToD[0] if ToD[0] is not None else "Unknown" for ToD in operators]
    selected_operator = st.sidebar.selectbox("Select Operator", operator_list)
    
    #Time of Day filter
    query_time_of_day = "SELECT DISTINCT TIME_OF_DAY FROM FAA_BIRD_STRIKES ORDER BY TIME_OF_DAY"
    cursor.execute(query_time_of_day)
    time_of_day = cursor.fetchall()
    #Create list for time of day
    time_of_day_list = ["All"] + [ToD[0] if ToD[0] is not None else "Unknown" for ToD in time_of_day]
    selected_time_of_day = st.sidebar.selectbox("Select Time of Day", time_of_day_list)

    #Aircraft filter
    query_aircraft = "SELECT DISTINCT AIRCRAFT FROM FAA_BIRD_STRIKES ORDER BY AIRCRAFT"
    cursor.execute(query_aircraft)
    aircraft = cursor.fetchall()
    #Create list of aircrafts
    aircraft_list = ["All"] + [ToD[0] if ToD[0] is not None else "Unknown" for ToD in aircraft]
    selected_aircraft = st.sidebar.selectbox("Select Aircraft", aircraft_list)

    #Initialize filters and parameters for dynamic query
    filters = []
    params = []

    #Build final query
    if selected_operator != "All":
        filters.append("OPERATOR = %s")
        params.append(selected_operator)
    if selected_time_of_day != "All":
        filters.append("TIME_OF_DAY = %s")
        params.append(selected_time_of_day)
    if selected_aircraft != "All":
        filters.append("AIRCRAFT = %s")
        params.append(selected_aircraft)
        
    if filters:
        where_clause = "WHERE " + " AND ".join(filters)
        query = f"{base_query} {where_clause} GROUP BY INCIDENT_YEAR, TIME_OF_DAY, OPERATOR, AIRCRAFT ORDER BY INCIDENT_YEAR"
    else:
        query = f"{base_query} GROUP BY INCIDENT_YEAR, TIME_OF_DAY, OPERATOR, AIRCRAFT ORDER BY INCIDENT_YEAR"

    #Execute the query
    cursor.execute(query, params)
    yearly_strikes = cursor.fetchall()
    
    #Create yearly strikes df
    yearly_strikes_df = pd.DataFrame(yearly_strikes, columns=["Year", "Time of Day", "Operator", "Aircraft", "Bird Strikes"])
    #Convert columns to numeric
    yearly_strikes_df["Year"] = pd.to_numeric(yearly_strikes_df["Year"], errors="coerce")
    yearly_strikes_df["Bird Strikes"] = pd.to_numeric(yearly_strikes_df["Bird Strikes"], errors="coerce")
 
    #Plot the data
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(yearly_strikes_df["Year"], yearly_strikes_df["Bird Strikes"], color="mediumpurple")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Wildlife Strikes", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)
        
###################################################################################################
#Configure Time of Day page.
elif selected == "Time":
    st.title("Number of Wildlife Strikes")
    
    #Query to get bird strikes grouped by time of day
    query_ToD = """
    SELECT 
        TIME_OF_DAY, COUNT(*) AS bird_strikes
    FROM FAA_BIRD_STRIKES
    GROUP BY TIME_OF_DAY
    ORDER BY bird_strikes ASC
    """

    #Execute query
    cursor.execute(query_ToD)
    time_of_day = cursor.fetchall()

    #Create time of day df, convert strikes to numeric
    time_of_day_df = pd.DataFrame(time_of_day, columns=["Time of Day", "Bird Strikes"])
    #Convert to numeric
    time_of_day_df["Bird Strikes"] = pd.to_numeric(time_of_day_df["Bird Strikes"], errors="coerce")
    #Convert na to unknown
    time_of_day_df['Time of Day'].fillna('Unknown', inplace=True)

    #Query to get bird_strikes by time
    query_time = """
    SELECT 
        HOUR(TIME), COUNT(*) AS bird_strikes
    FROM FAA_BIRD_STRIKES
    GROUP BY HOUR(TIME)
    ORDER BY bird_strikes ASC
    """
    
    #Execute query
    cursor.execute(query_time)
    time_hour = cursor.fetchall()

    #Create time  df, convert strikes to numeric
    time_df = pd.DataFrame(time_hour, columns=["Hour", "Bird Strikes"])
    #Convert to numeric
    time_df["Bird Strikes"] = pd.to_numeric(time_df["Bird Strikes"], errors="coerce")
    #Drop na values
    time_df = time_df.dropna()
    
    #Query to get bird strikes grouped by phase of flight
    query_phase_of_flight = """
    SELECT 
        PHASE_OF_FLIGHT, COUNT(*) AS bird_strikes
    FROM FAA_BIRD_STRIKES
    GROUP BY PHASE_OF_FLIGHT
    """

    #Execute query
    cursor.execute(query_phase_of_flight)
    phase_of_flight = cursor.fetchall()

    #Create phase of flight df, convert strikes to numeric
    phase_of_flight_df = pd.DataFrame(phase_of_flight, columns=["Phase of Flight", "Bird Strikes"])
    #Convert to numeric
    phase_of_flight_df["Bird Strikes"] = pd.to_numeric(phase_of_flight_df["Bird Strikes"], errors="coerce")
    #Convert na to unknown
    phase_of_flight_df['Phase of Flight'].fillna('En Route', inplace=True)
    #Group by phase of flight again because null values changes to Unknown
    phase_of_flight_df = phase_of_flight_df.groupby('Phase of Flight')['Bird Strikes'].sum().reset_index()
    #Order by bird strikes descending
    phase_of_flight_df = phase_of_flight_df.sort_values(by='Bird Strikes', ascending=False)

    #Create three columns for the bar charts side by side
    chart_col1, chart_col2, chart_col3 = st.columns(3)       
    with chart_col1:
        #Create number of bird strikes by time of day chart
        st.subheader("by Time of Day")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            time_of_day_df["Bird Strikes"], 
            labels=time_of_day_df["Time of Day"], 
            autopct="%1.1f%%", 
            startangle=90, 
            colors=plt.cm.tab10.colors
        )
        labeldistance=1.1,
        pctdistance=0.85 
        st.pyplot(fig)
        
    with chart_col2:
       #Create number of bird strikes by hour chart
       st.subheader("by Hour")
       fig, ax = plt.subplots(figsize=(8, 8))
       ax.bar(time_df["Hour"], time_df["Bird Strikes"], color="aquamarine")
       ax.set_xlabel("Hour", fontsize=12)
       ax.set_xticks(time_df["Hour"]) 
       ax.tick_params(axis="x", rotation=0)
       ax.set_ylabel("Number of Wildlife Strikes", fontsize=12)
       ax.tick_params(axis="x", rotation=45)
       st.pyplot(fig)
      
    with chart_col3:
        #Create number of bird strikes by phase of flight chart
        st.subheader("By Phase of Flight")
        fig, ax = plt.subplots()
        ax.barh(phase_of_flight_df["Phase of Flight"], phase_of_flight_df["Bird Strikes"], color='cornflowerblue')
        ax.set_xlabel("Number of Wildlife Strikes")
        ax.set_ylabel("Phase of Flight")
        ax.invert_yaxis()
        st.pyplot(fig)
       
###################################################################################################
#Configure Seasonality page 
elif selected == "Seasonality":
    st.title("Seasonality Analysis")
    
    #Query to monthly strike get data
    query_monthly_strikes = """
    SELECT 
        INCIDENT_MONTH, COUNT(*) AS bird_strikes
    FROM FAA_BIRD_STRIKES
    GROUP BY INCIDENT_MONTH
    ORDER BY INCIDENT_MONTH ASC
    """
    #Execute query
    cursor.execute(query_monthly_strikes)
    monthly_strikes = cursor.fetchall()

    #Create monthly strikes df
    monthly_strikes_df = pd.DataFrame(monthly_strikes, columns=["Month", "Bird Strikes"])
    #Convert to numeric
    monthly_strikes_df["Month"] = pd.to_numeric(monthly_strikes_df["Month"], errors="coerce")
    monthly_strikes_df["Bird Strikes"] = pd.to_numeric(monthly_strikes_df["Bird Strikes"], errors="coerce")
    
    #Query to quarterly strike get data
    query_quarterly_strikes = """
    SELECT 
        QUARTER(INCIDENT_DATE) as quarter, COUNT(*) AS bird_strikes
    FROM FAA_BIRD_STRIKES
    GROUP BY quarter
    ORDER BY quarter ASC
    """
    #Execute query
    cursor.execute(query_quarterly_strikes)
    quarterly_strikes = cursor.fetchall()

    #Create quarterly strikes df
    quarterly_strikes_df = pd.DataFrame(quarterly_strikes, columns=["Quarter", "Bird Strikes"])
    #Convert to numeric
    quarterly_strikes_df["Quarter"] = pd.to_numeric(quarterly_strikes_df["Quarter"], errors="coerce")
    quarterly_strikes_df["Bird Strikes"] = pd.to_numeric(quarterly_strikes_df["Bird Strikes"], errors="coerce")
    
    #Create two columns for the bar charts side by side
    chart_col1, chart_col2 = st.columns(2)    
    
    with chart_col1:
        #Plot monthly strikes bar chart
        st.subheader("Monthly Wildlife Strikes")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            monthly_strikes_df["Month"],
            monthly_strikes_df["Bird Strikes"],
            color="lightcoral",
            width=0.6
        )
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("Number of Wildlife Strikes", fontsize=12)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        st.pyplot(fig)
        
    with chart_col2:
        #Plot quarterly strikes bar chart
        st.subheader("Quarterly Wildlife Strikes")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            quarterly_strikes_df["Quarter"],
            quarterly_strikes_df["Bird Strikes"],
            color="darkorange",
            width=0.6
        )
        ax.set_xlabel("Quarter", fontsize=12)
        ax.set_ylabel("Number of Wildlife Strikes", fontsize=12)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        st.pyplot(fig)

###################################################################################################
#Configure Aircraft page.
elif selected == "Aircraft":
    st.title("Number of Wildlife Strikes")
    
    #Query to get bird strikes grouped by aircraft
    query_aircraft = """
    SELECT 
        AIRCRAFT, COUNT(*) AS bird_strikes
    FROM FAA_BIRD_STRIKES
    GROUP BY AIRCRAFT
    ORDER BY bird_strikes DESC
    LIMIT 25
    """

    #Execute query
    cursor.execute(query_aircraft)
    aircraft = cursor.fetchall()

    #Create phase of flight df, convert strikes to numeric
    aircraft_df = pd.DataFrame(aircraft, columns=["Aircraft", "Bird Strikes"])
    #Convert to numeric
    aircraft_df["Bird Strikes"] = pd.to_numeric(aircraft_df["Bird Strikes"], errors="coerce")
    
    #Query to get bird strikes grouped by engine type
    #Type of power A = reciprocating engine (piston): B = Turbojet: C = Turboprop: D = Turbofan: E = None (glider): F = Turboshaft (helicopter): Y = Other
    query_engine_class = """
    SELECT
        CASE 
              WHEN AC_CLASS = 'A' THEN 'Piston'
              WHEN AC_CLASS= 'B' THEN 'Turbojet'
              WHEN AC_CLASS = 'C' THEN 'Turboprop'
              WHEN AC_CLASS = 'D' THEN 'Turbofan'
              WHEN AC_CLASS = 'E' THEN 'Glider'
              WHEN AC_CLASS = 'F' THEN 'Helicopter'
              WHEN AC_CLASS = 'Y' THEN 'Other'
              ELSE 'Unknown'
        END AS ENGINE_CLASS,
        COUNT(*) AS bird_strikes
    FROM FAA_BIRD_STRIKES
    GROUP BY ENGINE_CLASS
    ORDER BY bird_strikes DESC
    """

    #Execute query
    cursor.execute(query_engine_class)
    engine_class = cursor.fetchall()

    #Create engine class df, convert strikes to numeric
    engine_class_df = pd.DataFrame(engine_class, columns=["Engine Class", "Bird Strikes"])
    #Convert to numeric
    engine_class_df["Bird Strikes"] = pd.to_numeric(engine_class_df["Bird Strikes"], errors="coerce")
    #Convert na to unknown
    engine_class_df['Engine Class'].fillna('Unknown', inplace=True)
    
    #Query to get bird strikes grouped by engine type
    query_num_engines = """
        SELECT 
            CASE 
                  WHEN NUM_ENGS::INT = 1 THEN 'One'
                  WHEN NUM_ENGS::INT = 2 THEN 'Two'
                  WHEN NUM_ENGS::INT = 3 THEN 'Three'
                  WHEN NUM_ENGS::INT = 4 THEN 'Four'
                  ELSE 'Unknown'
            END AS NUM_ENGINES,
            COUNT(*) AS bird_strikes
        FROM FAA_BIRD_STRIKES
        GROUP BY NUM_ENGINES
        ORDER BY NUM_ENGINES DESC
    """

    #Execute query
    cursor.execute(query_num_engines)
    num_engines = cursor.fetchall()

    #Create engine class df, convert strikes to numeric
    num_engines_df = pd.DataFrame(num_engines, columns=["Number of Engines", "Bird Strikes"])
    #Convert to numeric
    num_engines_df["Bird Strikes"] = pd.to_numeric(num_engines_df["Bird Strikes"], errors="coerce")
    #Convert na to unknown
    num_engines_df['Number of Engines'].fillna('Unknown', inplace=True)
    
    #Create two columns for the bar charts side by side
    chart_col1, chart_col2, chart_col3 = st.columns(3)  
    
    with chart_col1:
        st.subheader("by Aircraft Type")
        #Create bird strikes by aircraft chart
        fig, ax = plt.subplots(figsize=(10,5))
        ax.barh(aircraft_df["Aircraft"], aircraft_df["Bird Strikes"], color='slateblue')
        ax.set_xlabel("Number of Bird Strikes")
        ax.set_ylabel("Aircraft")
        ax.invert_yaxis()
        st.pyplot(fig)
    
    with chart_col2:
        st.subheader("by Engine Class")
        #Create bird strikes by engine class chart
        fig, ax = plt.subplots(figsize=(10,5))
        ax.bar(engine_class_df["Engine Class"], engine_class_df["Bird Strikes"], color='plum')
        ax.set_ylabel("Number of Bird Strikes")
        ax.set_xlabel("Engine Class")
        st.pyplot(fig)
        
    with chart_col3:
        st.subheader("by Number of Engines")
        #Create bird strikes by number of engines chart
        fig, ax = plt.subplots(figsize=(10,5))
        ax.bar(num_engines_df["Number of Engines"], num_engines_df["Bird Strikes"], color='midnightblue')
        ax.set_ylabel("Number of Wildlife Strikes")
        ax.set_xlabel("Number of Engines")
        #Customize the x-axis ticks and labels
        ordered_ticks = ['One', 'Two', 'Three', 'Four', 'Unknown']
        # Set the order of x-ticks explicitly
        ax.set_xticks(range(len(ordered_ticks)))
        ax.set_xticklabels(ordered_ticks)
        st.pyplot(fig)


###################################################################################################
#Configure Species page.
elif selected == "Species":
    st.title("Number of Wildlife Strikes by Species")
    
    #Query to get bird strikes grouped by phase of flight
    query_species = """
    SELECT 
        SPECIES, COUNT(*) AS bird_strikes
    FROM FAA_BIRD_STRIKES
    GROUP BY SPECIES
    ORDER BY bird_strikes DESC
    LIMIT 25
    """

    #Execute query
    cursor.execute(query_species)
    species = cursor.fetchall()

    #Create phase of flight df, convert strikes to numeric
    species_df = pd.DataFrame(species, columns=["Species", "Bird Strikes"])
    #Convert to numeric
    species_df["Bird Strikes"] = pd.to_numeric(species_df["Bird Strikes"], errors="coerce")
    
    #Create phase of flight chart
    fig, ax = plt.subplots(figsize=(10,5))
    ax.barh(species_df["Species"], species_df["Bird Strikes"], color='forestgreen')
    ax.set_xlabel("Number of Wildlife Strikes")
    ax.set_ylabel("Species")
    ax.invert_yaxis()
    st.pyplot(fig)
       
###################################################################################################
#Configure Scatter Plot page.    
elif selected == "Scatter Plot":
    #Sidebar filters
    #Add year range filter
    year_range = st.sidebar.slider("Select Year Range", min_value=1996, max_value=2025, value=(1996, 2025), step=1)
    start_year, end_year = year_range
    #Sidebar to select X variable
    x_variable = st.sidebar.selectbox(
        "Select X Variable",
        options=["Height", "Speed", "Distance"],
        index=0
    )
    
    #Define the corresponding column name in the database
    x_column_map = {
        "Height": "HEIGHT",
        "Speed": "SPEED",
        "Distance": "DISTANCE"
    }
    x_column = x_column_map[x_variable]  #Get selected column name
    
    #Define the query with dynamic X variable
    scatter_query = f"""
        SELECT 
            INCIDENT_DATE, TOTAL_COST, WARNED, {x_column}
        FROM FAA_BIRD_STRIKES
        WHERE TOTAL_COST IS NOT NULL 
        AND {x_column} IS NOT NULL
        AND YEAR(INCIDENT_DATE) BETWEEN %s AND %s
    """
    
    #Dynamic query
    filters = []
    params = []

    #Execute the query with year filtering
    cursor.execute(scatter_query, (start_year, end_year))
    scatter = cursor.fetchall()
    
    #Create scatter df
    scatter_df = pd.DataFrame(scatter, columns=["Date", "Total_Cost", "Warned", x_variable])
    #Make sure values are numeric
    scatter_df["Total_Cost"] = pd.to_numeric(scatter_df["Total_Cost"], errors="coerce")
    #Convert to numeric
    scatter_df[x_variable] = pd.to_numeric(scatter_df[x_variable], errors="coerce")
    #Apply log transformation
    scatter_df[f"Log_{x_variable}"] = np.log10(scatter_df[x_variable] + 1)
    scatter_df["Log_Cost"] = np.log10(scatter_df["Total_Cost"] + 1)
    #Replace null with Unknown
    scatter_df['Warned'] = scatter_df['Warned'].fillna('Unknown').astype('category')
    #Drop NAs
    scatter_df = scatter_df.dropna(subset=[f"Log_{x_variable}", "Log_Cost"])
    
    #Calculate Pearson's r
    pearson_r, _ = stats.pearsonr(scatter_df[f"Log_{x_variable}"], scatter_df["Log_Cost"])
    
    #Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
    data=scatter_df, 
    x=f"Log_{x_variable}", 
    y="Log_Cost", 
    hue="Warned",      #Color by 'Warned'
    ax=ax
    )
    #Set axis labels and title
    ax.set_xlabel(f"Log {x_variable}", fontsize=12)
    ax.set_ylabel("Log Cost", fontsize=12)
    ax.set_title(f"Scatter Plot: Log Cost vs Log {x_variable} | Pearson's r = {pearson_r:.2f}", fontsize=14)

    #Calculate Pearson's r for each 'Warned' category and display on plot
    for warned_value in scatter_df['Warned'].unique():
        subset = scatter_df[scatter_df['Warned'] == warned_value]
        pearson_r, _ = stats.pearsonr(subset[f"Log_{x_variable}"], subset["Log_Cost"])
        ax.text(
            0.95, 0.05 + 0.05 * list(scatter_df['Warned'].unique()).index(warned_value), 
            f"Pearson's r ({warned_value}) = {pearson_r:.2f}", 
            ha="left", va="top", transform=ax.transAxes, 
            fontsize=12, color="black"
        )
    
    st.pyplot(fig)

###################################################################################################
#Configure About page.
elif selected == "About":
    st.title("About")
    st.markdown("""
        Author: Vlad Lee 
        Feel free to check out my profile pages and GitHub!

        <p><a href='https://www.linkedin.com/in/vlad-lee' target='_blank'>LinkedIn</a></p>
        <p><a href='https://github.com/Vlad-Lee/Aviation-Wildlife-Strikes' target='_blank'>GitHub</a></p>
        <p>For questions or feedback, contact the author at 
        <a href='mailto:Vlad7984@gmail.com'>vlad7984@gmail.com</a>.</p>
    """, unsafe_allow_html=True)