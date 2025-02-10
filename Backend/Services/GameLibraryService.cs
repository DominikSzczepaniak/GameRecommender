using System.Text.RegularExpressions;
using GameRecommender.Data;
using GameRecommender.Interfaces;

namespace GameRecommender.Services;

public class GameLibraryService(IDatabaseHandler databaseHandler) : IGameLibrary
{
    private const string SteamPattern = @"https://steamcommunity\.com/profiles/([^/]+)/";
    private static readonly Regex SteamRegex = new Regex(SteamPattern);

    private string GetSteamId(string steamProfileLink)
    {
        Match match = SteamRegex.Match(steamProfileLink);
        if (match.Success)
        {
            return match.Groups[1].Value;
        }

        throw new ArgumentException("Invalid steam link or user not found");
    }

    private async Task<List<GameData>> GetSteamGamesFromXml(int userId)
    {
        string steamId = await GetUserSteamId(userId);
        return await SteamGameFetcher.GetSteamGamesFromXmlAsync(steamId);
    }

    private Task AddGamesToUserLibrary(int userId, List<GameData> gamesToAdd)
    {
        return new Task<int>(() => 1);
        //TODO create user games repository and add it into that. Later move repository into database
    }

    public async Task SetUserSteamProfile(int userId, string steamProfileLink)
    {
        var steamId = GetSteamId(steamProfileLink);
        await databaseHandler.SetUserSteamProfileId(userId, steamId);
        await AddGamesToUserLibrary(userId, await GetSteamGamesFromXml(userId));
    }

    private async Task<String> GetUserSteamId(int userId)
    {
        return await Task.FromResult("1"); //TODO
    }
}