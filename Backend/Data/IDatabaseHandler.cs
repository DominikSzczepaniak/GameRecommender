using GameRecommender.Models;

namespace GameRecommender.Data;

public interface IDatabaseHandler
{
    public Task<User?> LoginByUsername(string username, string password);
    public Task<bool> GameChosenInGallery(Guid userId);
    public Task RegisterUser(User user);
    public Task<User> UpdateUser(User user);
    public Task<bool> DeleteUser(User user);
    public Task SetUserSteamProfileId(Guid userId, string steamProfileId);
    public Task<string> GetUserSteamId(Guid userId);
    public Task AddGameToUserLibrary(UserGameDao userGameDao);
    public Task<List<UserGameLogic>> GetUserGames(Guid userId);
    public Task AddOpinionForUserAndGame(UserGameDao userGameDao);
    public Task AddAppIdToNameMapping(string appId, string name);
}