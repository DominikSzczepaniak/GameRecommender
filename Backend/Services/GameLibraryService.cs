using System.Text.RegularExpressions;
using GameRecommender.Data;
using GameRecommender.Interfaces;
using GameRecommender.Interfaces.Data;

namespace GameRecommender.Services;

public class GameLibraryService : IGameLibrary
{
    private const string SteamPattern = @"https://steamcommunity\.com/profiles/([^/]+)/";
    private static readonly Regex SteamRegex = new Regex(SteamPattern);
    private readonly IGameLibraryHandler _gameLibraryHandler;
    private readonly ISteamProfileHandler _steamProfileHandler;

    public GameLibraryService(IGameLibraryHandler gameLibraryHandler, ISteamProfileHandler steamProfileHandler)
    {
        _gameLibraryHandler = gameLibraryHandler;
        _steamProfileHandler = steamProfileHandler;
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
        gamesToAdd.ForEach(game => _gameLibraryHandler.AddGameToUserLibrary(game.ToDao(userId)));
    }

    public async Task SetUserSteamProfile(Guid userId, string steamProfileLink)
    {
        var steamId = GetSteamId(steamProfileLink);
        AddGamesToUserLibrary(userId, await GetSteamGamesFromXml(userId));
        await _steamProfileHandler.SetUserSteamProfileId(userId, steamId);
    }

    private async Task<String> GetUserSteamId(Guid userId)
    {
        return await _steamProfileHandler.GetUserSteamId(userId);
    }
}