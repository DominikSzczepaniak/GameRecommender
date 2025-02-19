using GameRecommender.Models;

namespace GameRecommender.Data;

public class SteamFetchedGameData
{
    public string AppId { get; set; }
    public string Name { get; set; }
    public double HoursOnRecord { get; set; }

    public SteamFetchedGameData(string appId, string name, double hoursOnRecord)
    {
        AppId = appId;
        Name = name;
        HoursOnRecord = hoursOnRecord;
    }

    public UserGameDao ToDao(Guid userId)
    {
        return new UserGameDao(userId, AppId, HoursOnRecord);
    }
}