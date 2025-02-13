using System.ComponentModel.DataAnnotations;

namespace GameRecommender.Models;


public class AppIdToName
{
    [Key]
    [MaxLength(64)]
    public string AppId { get; set; }
    [MaxLength(64)]
    public string Name { get; set; }
}