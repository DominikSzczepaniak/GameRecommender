using System.Text.RegularExpressions;
using GameRecommender.Data;
using GameRecommender.Interfaces;

namespace GameRecommender.Services;

public class GameLibraryService : IGameLibrary
{
    private const string SteamPattern = @"https://steamcommunity\.com/profiles/([^/]+)/";
    private static readonly Regex SteamRegex = new Regex(SteamPattern);
    private readonly IDatabaseHandler _databaseHandler;

    public GameLibraryService(IDatabaseHandler databaseHandler)
    {
        _databaseHandler = databaseHandler;
    }

    private string GetSteamId(string steamProfileLink)
    {
        Match match = SteamRegex.Match(steamProfileLink);
        if (match.Success)
        {
            return match.Groups[1].Value;
        }

        throw new ArgumentException("Invalid steam link or user not found");
    }

    private async Task<List<SteamFetchedGameData>> GetSteamGamesFromXml(Guid userId)
    {
        string steamId = await GetUserSteamId(userId);
        return await SteamGameFetcher.GetSteamGamesFromXmlAsync(steamId);
    }

    private void AddGamesToUserLibrary(Guid userId, List<SteamFetchedGameData> gamesToAdd)
    {
        gamesToAdd.ForEach(game => _databaseHandler.AddGameToUserLibrary(game.ToDao(userId)));
    }

    public async Task SetUserSteamProfile(Guid userId, string steamProfileLink)
    {
        var steamId = GetSteamId(steamProfileLink);
        AddGamesToUserLibrary(userId, await GetSteamGamesFromXml(userId));
        await _databaseHandler.SetUserSteamProfileId(userId, steamId);
    }

    private async Task<String> GetUserSteamId(Guid userId)
    {
        return await _databaseHandler.GetUserSteamId(userId);
    }
}