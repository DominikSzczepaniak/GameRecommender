using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
namespace GameRecommender.Models;

public class UserGameDao
{
    public Guid UserId { get; set; }
    public string AppId { get; set; }

    public bool? Opinion { get; set; } = null;// Null for no opinion, false for dislike, true for like
    public double? Playtime { get; set; } = null;

    public UserGameDao(Guid userId, string appId, double? playtime, bool? opinion = null)
    {
        UserId = userId;
        AppId = appId;
        Playtime = playtime;
        Opinion = opinion;
    }
}