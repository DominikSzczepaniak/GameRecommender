using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace GameRecommender.Models;

public class UserToSteamId
{
    [Key]
    [ForeignKey("User")]
    public Guid UserId { get; set; }

    [Required]
    [MaxLength(96)]
    public string SteamId { get; set; } = string.Empty;

    public User User { get; set; }
}