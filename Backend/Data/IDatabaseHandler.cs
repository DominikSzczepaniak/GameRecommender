using GameRecommender.Models;

namespace GameRecommender.Data;

public interface IDatabaseHandler
{
    public Task<User> LoginByUsername(string username, string password);
    public Task<User> RegisterUser(User user);
    public Task<User> UpdateUser(User user);
    public Task<bool> DeleteUser(User user);
    public Task SetUserSteamProfileId(int userId, string steamProfileId);
}