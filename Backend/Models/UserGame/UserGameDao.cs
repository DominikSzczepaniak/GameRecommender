using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
namespace GameRecommender.Models;

public class UserGameDao
{
    [Key]
    [ForeignKey("User")]
    public Guid UserId { get; set; }
    [MaxLength(64)]
    public string AppId { get; set; }

    public bool? Opinion { get; set; } = null;// Null for no opinion, false for dislike, true for like
    public double? Playtime { get; set; } = null;

    public User User { get; set; }

    public UserGameDao(Guid userId, string appId, double? playtime, bool? opinion = null)
    {
        UserId = userId;
        AppId = appId;
        Playtime = playtime;
        Opinion = opinion;
    }
}