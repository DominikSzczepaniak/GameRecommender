using GameRecommender.Models;

namespace GameRecommender.Interfaces.Data;

public interface IUserHandler
{
    Task<User?> LoginByUsername(string username, string password);
    Task RegisterUser(User user);
    Task<User> UpdateUser(User user);
    Task<bool> DeleteUser(User user);
}