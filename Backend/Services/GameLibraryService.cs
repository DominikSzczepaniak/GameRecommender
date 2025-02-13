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

    private async Task<List<GameData>> GetSteamGamesFromXml(Guid userId)
    {
        string steamId = await GetUserSteamId(userId);
        return await SteamGameFetcher.GetSteamGamesFromXmlAsync(steamId);
    }

    private void AddGamesToUserLibrary(Guid userId, List<GameData> gamesToAdd)
    {
        gamesToAdd.ForEach(game => databaseHandler.AddGameToUserLibrary(userId, game.AppId, game.HoursOnRecord));
    }

    public async Task SetUserSteamProfile(Guid userId, string steamProfileLink)
    {
        var steamId = GetSteamId(steamProfileLink);
        AddGamesToUserLibrary(userId, await GetSteamGamesFromXml(userId));
        await databaseHandler.SetUserSteamProfileId(userId, steamId);
    }

    private async Task<String> GetUserSteamId(Guid userId)
    {
        return await databaseHandler.GetUserSteamId(userId);
    }
}