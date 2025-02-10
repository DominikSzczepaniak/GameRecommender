using GameRecommender.Models;

namespace GameRecommender.Data;

public interface IDatabaseHandler
{
    public Task<User> GetUserByUsername(string username, string password);
    public Task<User> GetUserByEmail(string email, string password);
    public Task<User> RegisterUser(string username, string email, string password);
    public Task<User> UpdateUser(int id, string username, string email, string password);
    public Task<User> DeleteUser(string username, string email, string password);
    public Task SetUserSteamProfileId(int userId, string steamProfileId);
}