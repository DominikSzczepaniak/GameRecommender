using GameRecommender.Models;
using Microsoft.EntityFrameworkCore;
namespace GameRecommender.Data;

public class ApplicationDbContext : DbContext
{
    public ApplicationDbContext(DbContextOptions<ApplicationDbContext> options) : base(options) { }

    public DbSet<User> Users { get; set; }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<User>()
            .HasKey(u => u.Id);

        modelBuilder.Entity<UserToSteamId>()
            .HasOne(us => us.User)
            .WithOne()
            .HasForeignKey<UserToSteamId>(us => us.UserId)
            .OnDelete(DeleteBehavior.Cascade);

        modelBuilder.Entity<UserToSteamId>()
            .HasKey(us => us.UserId);

        modelBuilder.Entity<UserGameDao>()
            .HasKey(ug => new { ug.UserId, ug.AppId });

        modelBuilder.Entity<UserGameDao>()
            .HasOne(ug => ug.User)
            .WithMany()
            .HasForeignKey(ug => ug.UserId)
            .OnDelete(DeleteBehavior.Cascade);

        modelBuilder.Entity<AppIdToName>().HasKey(atn => new { atn.AppId });
    }
}