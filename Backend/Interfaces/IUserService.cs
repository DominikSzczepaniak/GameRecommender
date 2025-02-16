using GameRecommender.Models;

namespace GameRecommender.Interfaces;

public interface IUserService
{
    Task<User> LoginByUsername(string username, string password);
    Task RegisterUser(User user);
    Task<User> UpdateUser(User user); //update all data for given user id (only accessible for logged user)
    Task<bool> DeleteUser(User user);
}