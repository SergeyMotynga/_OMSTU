import { createRouter, createWebHistory } from 'vue-router'
import LaureatsView from '../views/LaureatsView.vue'
import PrizesView from '../views/PrizesView.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      redirect: '/laureats'
    },
    {
      path: '/laureats',
      name: 'Laureats',
      component: LaureatsView
    },
    {
      path: '/prizes',
      name: 'Prizes',
      component: PrizesView
    }
  ]
})

export default router

