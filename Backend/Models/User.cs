namespace GameRecommender.Models;

public class User
{
    public Guid Id { get; }
    public string Username { get; set; }
    public string Email { get; set; }
    public string Password { get; private set; }
    public List<int> PlayedGames { get; set; } //ints are app_id
    public Dictionary<int, bool> GamesOpinions { get; set; } //app_id -> False (dislike) / True (like)

    public User(Guid anId, string aUsername, string anEmail, string aPassword, List<int> thePlayedGames, Dictionary<int, bool> theGamesOpinions)
    {
        Id = anId;
        Username = aUsername;
        Email = anEmail;
        Password = aPassword;
        PlayedGames = thePlayedGames;
        GamesOpinions = theGamesOpinions;
    }
}