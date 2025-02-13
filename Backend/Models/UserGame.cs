using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
namespace GameRecommender.Models;

public class UserGame
{
    [Key]
    [ForeignKey("User")]
    public Guid UserId { get; set; }
    [MaxLength(64)]
    public string AppId { get; set; }
    public bool? Opinion { get; set; } // Null for no opinion, false for dislike, true for like
    public double Playtime;

    public User User { get; set; }

    public UserGame(Guid userId, string appId, double playtime, bool? opinion = null)
    {
        UserId = userId;
        AppId = appId;
        Playtime = playtime;
        Opinion = opinion;
    }
}