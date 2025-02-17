using Microsoft.EntityFrameworkCore;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace GameRecommender.Models;

public class User
{
    [Key]
    public Guid Id { get; set; }
    [MaxLength(64)]
    public string Username { get; set; }
    [MaxLength(64)]
    public string Email { get; set; }
    [MaxLength(32)]
    public string Password { get; set; }
    public User () {}

    public User(Guid anId, string aUsername, string anEmail, string aPassword)
    {
        Id = anId;
        Username = aUsername;
        Email = anEmail;
        Password = aPassword;
    }
}