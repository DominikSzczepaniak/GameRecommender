using System.Text.RegularExpressions;
using Data;
using GameRecommender.Interfaces;

namespace GameRecommender.Services;

public class GameLibraryService(IDatabaseHandler databaseHandler) : IGameLibrary
{
    private static string steamPattern = @"https://steamcommunity\.com/profiles/([^/]+)/";
    private static Regex steamRegex = new Regex(steamPattern);

    private string GetSteamId(string steamProfileLink)
    {
        Match match = steamRegex.Match(steamProfileLink);
        if (match.Success)
        {
            return match.Groups[1].Value;
        }

        throw new ArgumentException("Invalid steam link or user not found");
    }

    private List<string> GetSteamGamesFromXML(int userId, string steamId)
    {
        string link = $"https://steamcommunity.com/profiles/{steamId}/games?tab=all&xml=1";
        return new List<string>(); //TODO make request to link, parse XML, return all games as List
    }

    private Task AddGamesToUserLibrary(int userId, List<string> gamesToAdd)
    {
        return new Task<int>(() => 1);
        //TODO create user games repository and add it into that. Later move repository into database
    }

    public async Task SetUserSteamProfile(int userId, string steamProfileLink)
    {
        var steamId = GetSteamId(steamProfileLink);
        await databaseHandler.SetUserSteamProfileId(userId, steamId);
        await AddGamesToUserLibrary(userId, GetSteamGamesFromXML(userId, steamId));
    }
}