using GameRecommender.Models;

namespace Data;

public interface IDatabaseHandler
{
    public Task<User> LoginByUsername(string username, string password);
    public Task<User> RegisterUser(string username, string email, string password);
    public Task<User> UpdateUser(int id, string username, string email, string password);
    public Task<User> DeleteUser(string username, string email, string password);
    public Task SetUserSteamProfileId(int userId, string steamProfileId);
}