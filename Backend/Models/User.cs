namespace GameRecommender.Models;

public class User(string username, string email, string password)
{
    public string Username { get; set; } = username;
    public string Email { get; set; } = email;
    public string Password { get; set; } = password;
}