using GameRecommender.Models;

namespace GameRecommender.Data;

public interface IDatabaseHandler
{
    public Task<User?> LoginByUsername(string username, string password);
    public Task<User> RegisterUser(User user);
    public Task<User> UpdateUser(User user);
    public Task<bool> DeleteUser(User user);
    public Task SetUserSteamProfileId(Guid userId, string steamProfileId);
    public Task<string> GetUserSteamId(Guid userId);
    public Task AddGameToUserLibrary(Guid userId, string appId, double playtime, bool? opinion = null);
    public Task<List<GameData>> GetUserGames(Guid userId);
    public Task AddOpinionForUserAndGame(Guid userId, string appId, bool opinion);
    public Task AddAppIdToNameMapping(string appId, string name);
}