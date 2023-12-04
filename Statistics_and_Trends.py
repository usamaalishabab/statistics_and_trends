import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def reading_and_manipulating_data(filename):
    '''
    Read data from a CSV file, perform preprocessing, and return years as columns and countries as an index.
    
    Parameters:
        filename (str): The name of the CSV file.
        
    Returns:
        pd.DataFrame: Transposed DataFrame with years as columns and countries as index.
        pd.DataFrame: Original data DataFrame.
    '''
    # Read csv file and set variables
    df = pd.read_csv(filename, skiprows=4)

    # Drop unnecessary columns
    columns_to_drop = ['Country Code', 'Indicator Code', 'Unnamed: 67']
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Transpose the DataFrame
    transposed_df = df.set_index(['Country Name', 'Indicator Name']).T

    return transposed_df, df


def filter_indicator_data(indicator, data):
    '''
    Extract the data for a specific indicator from the World Bank dataset.
    
    Parameters:
        indicator (str): The name of the indicator.
        data (pd.DataFrame): The original data DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame containing data for the specified indicator.
    '''
    return data[data['Indicator Name'] == indicator]


def line_graph(data, indicator_name, y_label):
    '''
    Create a line graph for specified countries over the years for a given indicator.

    Parameters:
        data (pd.DataFrame): Indicator data for specified countries.
        indicator_name (str): Name of the indicator.
        y_label (str): Label for the y-axis.
    '''
    countries = ['Pakistan', 'India', 'China',
                 'Spain', 'Bangladesh', 'United States']
    years = [str(year) for year in range(2000, 2023)]

    data = data[data['Country Name'].isin(countries)]
    data = data[['Country Name'] + years]
    data.set_index('Country Name', inplace=True)

    plt.figure(figsize=(10, 6))
    for country in data.index:
        plt.plot(data.columns, data.loc[country], label=country, marker='o')

    plt.xlabel('Year')
    plt.ylabel(y_label)
    plt.title(indicator_name.upper())
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend()
    plt.show()


def bar_plot(data, indicator_name, y_label):
    '''
    Create a bar plot for specified countries for a given indicator.

    Parameters:
        data (pd.DataFrame): Indicator data for specified countries.
        indicator_name (str): Name of the indicator.
        y_label (str): Label for the y-axis.
    '''
    countries = ['Pakistan', 'India', 'China',
                 'Spain', 'Bangladesh', 'United States']
    years = [str(year) for year in range(2000, 2022)]

    data = data[data['Country Name'].isin(countries)]
    data = data[['Country Name'] + years]
    data.set_index('Country Name', inplace=True)

    plt.figure(figsize=(10, 6))

    # Set width of each bar
    bar_width = 0.15

    for i, country in enumerate(data.index):
        # Calculate the position for each bar
        position = [pos + i * bar_width for pos in range(len(data.columns))]
        plt.bar(position, data.loc[country], width=bar_width, label=country)

    # Calculate the position for x-ticks (centered between the bars)
    tick_positions = [pos + 0.5 * bar_width *
                      (len(data.index) - 1) for pos in range(len(data.columns))]

    plt.xlabel('Year')
    plt.ylabel(y_label)
    plt.title(indicator_name.upper())
    plt.xticks(tick_positions, data.columns, rotation=45, ha='right')
    plt.tight_layout()
    plt.legend()
    plt.show()


def heatmap(data, name):
    '''
    Display a heatmap graph showing the correlation between different indicators causing global warming.
    
    Parameters:
        data (pd.DataFrame): Original data DataFrame.
        name (str): Name of the country for which the heatmap is generated.
    '''
    # Get country-specific data
    country_data = data[data['Country Name'] == name]

    # Define an empty DataFrame
    indicator_data = pd.DataFrame()

    # List of indicators to include in the heatmap
    indicators_to_include = [
        "Access to electricity (% of population)",
        "CO2 emissions (kt)",
        "Electric power consumption (kWh per capita)",
        "Population, total",
        "Forest area (sq. km)"
    ]

    # Populate the DataFrame with indicator data
    for indicator_name in indicators_to_include:
        indicator_data[indicator_name] = country_data[country_data['Indicator Name']
                                                      == indicator_name].iloc[:, 2:].values.flatten()

    # Drop unnecessary columns and reset index
    indicator_data = indicator_data.drop(
        ['Country Name', 'Indicator Name'], errors='ignore').reset_index(drop=True)

    # Plotting the data
    ax = plt.axes()

    # Use seaborn to plot heatmap
    sns.heatmap(indicator_data.corr(), cmap="YlGnBu", annot=True, ax=ax)

    # Adding title
    ax.set_title(name)

    # Show the plot
    plt.show()


# Main Function
if __name__ == "__main__":
    # Set Variables
    years, countries = reading_and_manipulating_data(
        'API_19_DS2_en_csv_v2_4756035.csv')

    # Draw line graph
    indicator_population = filter_indicator_data(
        "Population, total", countries)
    line_graph(indicator_population, 'Population, total', 'Population')

    # Draw line graph
    indicator_electricity = filter_indicator_data(
        "Access to electricity (% of population)", countries)
    line_graph(indicator_electricity, ' Access to electricity (% of population)',
               'Access to electricity (% of population)')

    # Draw bar plot
    indicator_co2 = filter_indicator_data("CO2 emissions (kt)", countries)
    bar_plot(indicator_co2, "CO2 emissions (kt)", 'CO2 emissions (kt)')

    # # Draw bar plot
    indicator_forest = filter_indicator_data("Forest area (sq. km)", countries)
    bar_plot(indicator_forest, "Forest area (sq. km)", 'Forest area (sq. km)')

    # Draw heatmap
    heatmap(countries, 'China')
    heatmap(countries, 'United States')
    heatmap(countries, 'India')
