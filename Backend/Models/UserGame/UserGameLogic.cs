namespace GameRecommender.Models;

public class UserGameLogic
{
    public string AppId { get; set; }
    public string? Name { get; set; }
    public double? Playtime { get; set; } = null;
    public bool? Opinion { get; set; } = null;

    public UserGameLogic(string appId, double? playtime, bool? opinion, string? name = null)
    {
        AppId = appId;
        Playtime = playtime;
        Opinion = opinion;
        Name = name;
    }

    public UserGameDao ToDao(Guid userId)
    {
        return new UserGameDao(userId, AppId, Playtime, Opinion);
    }
}