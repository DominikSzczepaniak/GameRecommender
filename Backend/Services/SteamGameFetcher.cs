using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Threading.Tasks;
using System.Xml.Linq;
using GameRecommender.Data; // Make sure this namespace is correct
using System.Globalization;
using System.Text.RegularExpressions;

namespace GameRecommender.Services;

public static class SteamGameFetcher
{
    private static async Task<string> GetXmlDataAsync(string link)
    {
        using (HttpClient client = new HttpClient())
        {
            try
            {
                HttpResponseMessage response = await client.GetAsync(link);
                response.EnsureSuccessStatusCode();
                return await response.Content.ReadAsStringAsync();
            }
            catch (HttpRequestException ex)
            {
                Console.WriteLine($"Error making request: {ex.Message}");
                return null;
            }
        }
    }

    public static async Task<List<SteamFetchedGameData>> GetSteamGamesFromXmlAsync(string steamId)
    {
        string link = $"https://steamcommunity.com/profiles/{steamId}/games?tab=all&xml=1";
        string xmlData = await GetXmlDataAsync(link);

        if (xmlData == null)
        {
            return new List<SteamFetchedGameData>();
        }

        try
        {
            XDocument doc = XDocument.Parse(xmlData);
            List<SteamFetchedGameData> games = new List<SteamFetchedGameData>();

            foreach (XElement gameElement in doc.Descendants("game"))
            {
                string appId = gameElement.Element("appID")?.Value ?? "";

                string name = gameElement.Element("name")?.Value ?? "";

                double hoursOnRecord = 0.0;

                XElement hoursElement = gameElement.Element("hoursOnRecord");
                if (hoursElement != null)
                {
                    string hoursString = hoursElement.Value.Trim();
                    hoursString = hoursString.Replace('\u00A0', ' '); //Handle no-break space
                    hoursString = hoursString.Replace(',', '.');

                    hoursString = Regex.Replace(hoursString, @"[.,\s]", ""); //Removes . , and spaces

                    if (double.TryParse(hoursString, NumberStyles.Float, CultureInfo.InvariantCulture, out double hours))
                    {
                        hoursOnRecord = hours;
                    }
                    else
                    {
                        Console.WriteLine($"Error parsing hours: {hoursString}");
                        hoursOnRecord = 0;
                    }
                }
                else
                {
                    hoursOnRecord = 0;
                }

                games.Add(new SteamFetchedGameData(appId, name, hoursOnRecord));
            }

            return games;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error parsing XML: {ex.Message}");
            return new List<SteamFetchedGameData>();
        }
    }
}