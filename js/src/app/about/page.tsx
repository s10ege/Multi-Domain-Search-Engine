import Link from "next/link";
import styles from "./about.module.css";

export default function AboutPage() {
  return (
    <div className={styles.page}>
      <main className={styles.card}>
        <div className={styles.brand}>SemantikML</div>
        <h1 className={styles.title}>About</h1>
        <p className={styles.subtitle}>Placeholder copy. Replace with your final text.</p>
        <div className={styles.body}>
          <p>
            This page is ready for your About content. Share your narrative, mission,
            and product story here.
          </p>
        </div>
        <Link className={styles.backLink} href="/">
          Back to search
        </Link>
      </main>
    </div>
  );
}
